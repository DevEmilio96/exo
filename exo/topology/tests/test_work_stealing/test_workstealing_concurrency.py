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

    async def test_concurrent_work_stealing(self):
        print("\nConcurrent Work Stealing")

        # Create only one shard to be processed and assign it to node1
        shards = [Shard("model", 0, 9, 100)]
        
        # Call partition to initialize work queue and node capabilities
        self.work_stealing_strategy.partition(self.topology)
        
        # Add a single shard to node1
        self.work_stealing_strategy.add_shards(shards)

        # Simulate node2 and node3 attempting to steal work from node1 at the same time
        async def node2_steals():
            return await self.work_stealing_strategy.get_work("node2")

        async def node3_steals():
            return await self.work_stealing_strategy.get_work("node3")

        # Run both stealing attempts concurrently
        task1 = asyncio.create_task(node2_steals())
        task2 = asyncio.create_task(node3_steals())

        # Wait for both tasks to complete
        result1, result2 = await asyncio.gather(task1, task2)

        # Check results to ensure only one node successfully steals work at a time
        if result1 is not None and result2 is not None:
            print(f"Both nodes stole work concurrently. Node2: {result1}, Node3: {result2}")
        elif result1 is not None:
            print(f"Node2 successfully stole work: {result1}")
        elif result2 is not None:
            print(f"Node3 successfully stole work: {result2}")
        else:
            print("Neither node managed to steal work.")

        # Assert that only one of the two nodes stole work
        self.assertTrue(result1 is not None or result2 is not None, "At least one node should have stolen work.")
        self.assertFalse(result1 is not None and result2 is not None, "Only one node should have successfully stolen work.")

if __name__ == '__main__':
    unittest.main()
