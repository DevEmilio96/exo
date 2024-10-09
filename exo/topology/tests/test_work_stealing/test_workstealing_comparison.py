import unittest
import asyncio
from typing import List, Dict
from exo.topology.topology import Topology
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
from exo.topology.work_stealing_partitioning_strategy import WorkStealingPartitioningStrategy
from exo.inference.shard import Shard

class MockNode:
    def __init__(self, node_id: str, capabilities: DeviceCapabilities):
        self.node_id = node_id
        self.capabilities = capabilities
        self.work_done = 0
        self.is_active = True

    async def process_shard(self, shard: Shard):
        if not self.is_active:
            raise Exception(f"Node {self.node_id} is not active")
        shard_size = (shard.end_layer - shard.start_layer + 1) / shard.n_layers
        processing_time = shard_size / (self.capabilities.flops.fp16 / 1e12) * 1e2
        await asyncio.sleep(processing_time)
        self.work_done += 1

class TestImprovedWorkStealingStrategy(unittest.IsolatedAsyncioTestCase):
    
    async def asyncSetUp(self):
        self.topology = Topology()
        
        self.nodes = {
            "node1": MockNode("node1", DeviceCapabilities(
                model="NVIDIA RTX 4090", chip="RTX 4090", memory=24000,
                flops=DeviceFlops(fp32=82.58e12, fp16=165.16e12, int8=330.32e12),
                power_usage=300
            )),
            "node2": MockNode("node2", DeviceCapabilities(
                model="NVIDIA RTX 3080", chip="RTX 3080", memory=10000,
                flops=DeviceFlops(fp32=29.77e12, fp16=59.55e12, int8=119.1e12),
                power_usage=250
            )),
            "node3": MockNode("node3", DeviceCapabilities(
                model="NVIDIA RTX 3060", chip="RTX 3060", memory=12000,
                flops=DeviceFlops(fp32=13e12, fp16=26e12, int8=52e12),
                power_usage=200
            )),
            "node4": MockNode("node4", DeviceCapabilities(
                model="NVIDIA RTX 3050", chip="RTX 3050", memory=8000,
                flops=DeviceFlops(fp32=9.1e12, fp16=18.2e12, int8=36.4e12),
                power_usage=150
            )),
            "node5": MockNode("node5", DeviceCapabilities(
                model="NVIDIA GTX 1660", chip="GTX 1660", memory=6000,
                flops=DeviceFlops(fp32=5e12, fp16=10e12, int8=20e12),
                power_usage=100
            ))
        }

        for node_id, node in self.nodes.items():
            self.topology.update_node(node_id, node.capabilities)

        # Setup topology connections
        self.topology.add_edge("node1", "node2", latency=5, throughput=1000)
        self.topology.add_edge("node2", "node3", latency=12, throughput=500)
        self.topology.add_edge("node3", "node4", latency=15, throughput=750)
        self.topology.add_edge("node4", "node5", latency=20, throughput=500)
        self.topology.add_edge("node1", "node5", latency=25, throughput=400)

        self.ring_strategy = RingMemoryWeightedPartitioningStrategy()
        self.work_stealing_strategy = WorkStealingPartitioningStrategy()

    def split_shards(self, shards: List[Shard], max_shard_size: int) -> List[Shard]:
        new_shards = []
        for shard in shards:
            shard_size = shard.end_layer - shard.start_layer + 1
            if shard_size > max_shard_size:
                num_subshards = (shard_size + max_shard_size - 1) // max_shard_size
                subshard_size = shard_size // num_subshards
                for i in range(num_subshards):
                    start_layer = shard.start_layer + i * subshard_size
                    end_layer = min(start_layer + subshard_size - 1, shard.end_layer)
                    new_shards.append(Shard("model", start_layer, end_layer, shard.n_layers))
            else:
                new_shards.append(shard)
        return new_shards

    async def run_simulation(self, strategy, shards: List[Shard]) -> float:
        start_time = asyncio.get_event_loop().time()
        
        partitions = strategy.partition(self.topology)
        if isinstance(strategy, WorkStealingPartitioningStrategy):
            strategy.add_shards(shards)
            tasks = [self.process_node_work_stealing(node_id, strategy) for node_id in self.nodes]
        else:
            node_shards = {node_id: [] for node_id in self.nodes}
            for shard in shards:
                shard_fraction = (shard.start_layer + shard.end_layer + 1) / (2 * shard.n_layers)
                node_id = next(
                    p.node_id for p in partitions
                    if p.start <= shard_fraction < p.end
                )
                node_shards[node_id].append(shard)
            tasks = [self.process_node_shards(node_id, node_shards[node_id]) for node_id in self.nodes]

        await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = asyncio.get_event_loop().time()
        return end_time - start_time

    async def process_node_work_stealing(self, node_id: str, strategy: WorkStealingPartitioningStrategy):
        node = self.nodes[node_id]
        while True:
            shard = await strategy.get_work(node_id)
            if shard is None:
                break
            await node.process_shard(shard)

    async def process_node_shards(self, node_id: str, shards: List[Shard]):
        node = self.nodes[node_id]
        for shard in shards:
            await node.process_shard(shard)

    async def test_realistic_workload(self):
        print("\nRealistic Workload")
        shards = [Shard("model", i * 5, (i + 1) * 5 - 1, 100) for i in range(20)]
        shards += [Shard("model", 0, 49, 100)] * 5  # Add some large shards

        shards = self.split_shards(shards, max_shard_size=10)
        
        ring_time = await self.run_simulation(self.ring_strategy, shards)
        ws_time = await self.run_simulation(self.work_stealing_strategy, shards)

        print(f"Realistic Workload - Ring: {ring_time:.2f}s, Work Stealing: {ws_time:.2f}s")
        self.assertLess(ws_time, ring_time)

    async def test_node_failure(self):
        print("\nNode Failure")
        shards = [Shard("model", i * 5, (i + 1) * 5 - 1, 100) for i in range(20)]

        async def simulate_node_failure():
            await asyncio.sleep(0.5)
            self.nodes["node3"].is_active = False

        async def run_with_failure(strategy):
            failure_task = asyncio.create_task(simulate_node_failure())
            sim_task = asyncio.create_task(self.run_simulation(strategy, shards))
            await asyncio.gather(failure_task, sim_task)
            return sim_task.result()

        ring_time = await run_with_failure(self.ring_strategy)

        self.nodes["node3"].is_active = True

        ws_time = await run_with_failure(self.work_stealing_strategy)

        print(f"Node Failure - Ring: {ring_time:.2f}s, Work Stealing: {ws_time:.2f}s")
        self.assertLess(ws_time, ring_time)

    async def test_heterogeneous_performance(self):
        print("\nHeterogeneous Performance")
        shards = [Shard("model", i * 2, (i + 1) * 2 - 1, 100) for i in range(50)]

        ring_time = await self.run_simulation(self.ring_strategy, shards)
        ws_time = await self.run_simulation(self.work_stealing_strategy, shards)

        print(f"Heterogeneous Performance - Ring: {ring_time:.2f}s, Work Stealing: {ws_time:.2f}s")
        self.assertLess(ws_time, ring_time)

    async def test_power_efficiency(self):
        print("\nPower Efficiency")
        shards = [Shard("model", i * 2, (i + 1) * 2 - 1, 100) for i in range(50)]

        async def run_and_measure_power(strategy):
            start_time = asyncio.get_event_loop().time()
            await self.run_simulation(strategy, shards)
            end_time = asyncio.get_event_loop().time()
            total_power = sum(
                node.capabilities.power_usage * (end_time - start_time) for node in self.nodes.values()
            )
            return total_power

        ring_power = await run_and_measure_power(self.ring_strategy)
        ws_power = await run_and_measure_power(self.work_stealing_strategy)

        print(f"Power Efficiency - Ring: {ring_power:.2f}W, Work Stealing: {ws_power:.2f}W")
        self.assertLess(ws_power, ring_power)
    

if __name__ == '__main__':
    unittest.main()
