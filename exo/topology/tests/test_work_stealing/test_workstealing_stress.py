import asyncio
import random
from typing import List
import unittest
from exo.inference.shard import Shard
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
from exo.topology.topology import Topology
from exo.topology.work_stealing_partitioning_strategy import WorkStealingPartitioningStrategy

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

class TestWorkStealingStress(unittest.IsolatedAsyncioTestCase):
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
                # Check if there is any work left globally
                if strategy.is_work_remaining():
                    # Wait for a short period before trying again
                    await asyncio.sleep(0.1)
                    continue
                else:
                    break  # No more work anywhere
            await node.process_shard(shard)

    async def process_node_shards(self, node_id: str, shards: List[Shard]):
        node = self.nodes[node_id]
        for shard in shards:
            await node.process_shard(shard)
                
    async def test_stress(self):
        print("\nStress Test")
        
        # Create a large number of shards with varying sizes
        shards = []
        for _ in range(50):  # 1000 shards
            start = random.randint(0, 90)
            end = start + random.randint(1, 10)
            shards.append(Shard("model", start, end, 100))

        # Function to continuously add new work
        async def add_work_continuously():
            # Wait until active_nodes is populated
            while not self.work_stealing_strategy.active_nodes:
                await asyncio.sleep(0.1)
            total_shards_added = 0
            max_shards = 500  # Increase the limit on the total number of shards
            while total_shards_added < max_shards:
                new_shards = [Shard("model", random.randint(0, 6), random.randint(6, 16), 32) for _ in range(16)]
                self.work_stealing_strategy.add_shards(new_shards)
                total_shards_added += len(new_shards)
                await asyncio.sleep(1)

        # Function to simulate random node failures
        async def random_node_failures():
            for _ in range(10):  # 10 random failures
                node = random.choice(list(self.nodes.values()))
                node_id = node.node_id
                node.is_active = False
                print(f"Node {node_id} has failed.")
                await asyncio.sleep(0.5)
                node.is_active = True
                await asyncio.sleep(0.5)

        # Run the simulation with work stealing
        async def run_stress_test():
            add_work_task = asyncio.create_task(add_work_continuously())
            failure_task = asyncio.create_task(random_node_failures())
            sim_task = asyncio.create_task(self.run_simulation(self.work_stealing_strategy, shards))
            await asyncio.gather(add_work_task, failure_task, sim_task)
            return sim_task.result()

        start_time = asyncio.get_event_loop().time()
        await run_stress_test()
        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time

        # Calculate and print statistics
        total_work_done = sum(node.work_done for node in self.nodes.values())
        work_distribution = {node_id: node.work_done for node_id, node in self.nodes.items()}

        print(f"Stress Test completed in {total_time:.2f} seconds")
        print(f"Total work done: {total_work_done}")
        print(f"Work distribution: {work_distribution}")

        # Ensure all nodes have done some work
        nodes_with_work = sum(1 for work in work_distribution.values() if work > 0)
        self.assertGreaterEqual(nodes_with_work, len(self.nodes) - 1, "Almost all nodes should have done some work")

        # Verify that the work distribution is not too imbalanced
        work_counts = [work for work in work_distribution.values() if work > 0]
        if work_counts:
            max_work = max(work_counts)
            min_work = min(work_counts)
            imbalance_ratio = max_work / min_work if min_work > 0 else float('inf')
            self.assertLessEqual(imbalance_ratio, 3, "Work distribution should not be too imbalanced")
        else:
            self.fail("No nodes did any work")

if __name__ == '__main__':
    unittest.main()
