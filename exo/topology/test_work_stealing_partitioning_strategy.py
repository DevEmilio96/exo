import unittest
import asyncio
from typing import List, Optional, Tuple, Dict
from exo.topology.topology import Topology
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
from exo.topology.work_stealing_partitioning_strategy import WorkStealingPartitioningStrategy
from exo.inference.shard import Shard

class MockNode:
    def __init__(self, node_id: str, capabilities: DeviceCapabilities, topology: Topology):
        self.node_id = node_id
        self.capabilities = capabilities
        self.work_done = 0
        self.is_active = True
        self.topology = topology

    async def process_shard(self, shard: Shard, previous_node_id: str = None):
        if not self.is_active:
            raise Exception(f"Node {self.node_id} is not active")

        # Simulate processing time based on FLOPS and inter-node latency
        shard_size = (shard.end_layer - shard.start_layer + 1) / shard.n_layers
        compute_time = shard_size / (self.capabilities.flops.fp16 / 1e12) * 1e2  # Scaled compute time

        # Simulate network latency
        if previous_node_id:
            latency = self.topology.get_latency(previous_node_id, self.node_id)
        else:
            latency = 0  # No latency for the first node

        await asyncio.sleep(compute_time + latency)
        self.work_done += 1

class TestPartitioningStrategies(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.topology = Topology()
        self.nodes = {}
        self.ring_strategy = RingMemoryWeightedPartitioningStrategy()
        self.work_stealing_strategy = WorkStealingPartitioningStrategy()

    async def setup_scenario(self, node_configs, edge_configs):
        self.nodes = {}
        self.topology = Topology()
        for node_id, capabilities in node_configs.items():
            self.nodes[node_id] = MockNode(node_id, capabilities, self.topology)
            self.topology.update_node(node_id, capabilities)

        for edge in edge_configs:
            self.topology.add_edge(edge['from'], edge['to'], latency=edge['latency'])

    async def run_simulation(self, strategy, shards: List[Shard]) -> Tuple[float, float]:
        start_time = asyncio.get_event_loop().time()

        partitions = strategy.partition(self.topology)
        if isinstance(strategy, WorkStealingPartitioningStrategy):
            for shard in shards:
                await strategy.add_work(list(self.nodes.keys())[0], shard)  # Add all shards to the first node
            tasks = [self.process_node_work_stealing(node_id, strategy) for node_id in self.nodes]
        else:
            node_shards = {node_id: [] for node_id in self.nodes}
            for shard in shards:
                shard_fraction = (shard.start_layer + shard.end_layer + 1) / (2 * shard.n_layers)
                partition = next(
                    p for p in partitions
                    if p.start <= shard_fraction < p.end
                )
                node_shards[partition.node_id].append((shard, None))  # No previous node for first shard

            tasks = [self.process_node_shards(node_id, node_shards[node_id]) for node_id in self.nodes]

        await asyncio.gather(*tasks, return_exceptions=True)

        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time

        # Calculate latency to first token (processing time of first shard)
        latency_to_first_token = await self.calculate_latency_to_first_token(strategy, shards[0])

        return total_time, latency_to_first_token

    async def calculate_latency_to_first_token(self, strategy, first_shard: Shard) -> float:
        # Simulate processing of the first shard to measure latency to first token
        node_id = self.get_node_for_shard(strategy, first_shard)
        node = self.nodes[node_id]
        shard_size = (first_shard.end_layer - first_shard.start_layer + 1) / first_shard.n_layers
        compute_time = shard_size / (node.capabilities.flops.fp16 / 1e12) * 1e2

        latency = 0  # No previous node for the first shard

        return compute_time + latency

    def get_node_for_shard(self, strategy, shard: Shard) -> str:
        partitions = strategy.partition(self.topology)
        shard_fraction = (shard.start_layer + shard.end_layer + 1) / (2 * shard.n_layers)
        partition = next(
            p for p in partitions
            if p.start <= shard_fraction < p.end
        )
        return partition.node_id

    async def process_node_work_stealing(self, node_id: str, strategy: WorkStealingPartitioningStrategy):
        node = self.nodes[node_id]
        while node.is_active:
            shard = await strategy.get_work(node_id)
            if shard is None:
                if strategy.is_work_remaining():
                    await asyncio.sleep(0.1)
                    continue
                else:
                    break
            await node.process_shard(shard)

    async def process_node_shards(self, node_id: str, shard_tuples: List[Tuple[Shard, Optional[str]]]):
        node = self.nodes[node_id]
        for shard, previous_node_id in shard_tuples:
            await node.process_shard(shard, previous_node_id=previous_node_id)

    # Test Scenario 1: Homogeneous Nodes with Uniform Latency
    async def test_homogeneous_uniform_latency(self):
        print("\nTest Scenario 1: Homogeneous Nodes with Uniform Latency")
        node_configs = {
            f"node{i}": DeviceCapabilities(
                model="Generic GPU",
                chip="GenericChip",
                memory=16000,
                flops=DeviceFlops(fp32=10e12, fp16=20e12, int8=40e12),
                power_usage=200
            ) for i in range(1, 6)
        }
        edge_configs = [
            {'from': f"node{i}", 'to': f"node{j}", 'latency': 0.01}
            for i in range(1, 6) for j in range(i+1, 6)
        ]
        await self.setup_scenario(node_configs, edge_configs)

        total_layers = 100
        shards = [Shard("model", 0, total_layers - 1, total_layers)]

        ring_total_time, ring_latency = await self.run_simulation(self.ring_strategy, shards)
        ws_total_time, ws_latency = await self.run_simulation(self.work_stealing_strategy, shards)

        print(f"Ring Strategy - Total Time: {ring_total_time:.4f}s, Latency: {ring_latency:.4f}s")
        print(f"Work Stealing Strategy - Total Time: {ws_total_time:.4f}s, Latency: {ws_latency:.4f}s")

        tolerance = 0.01
        self.assertLessEqual(ws_total_time, ring_total_time * (1 + tolerance))
        self.assertLessEqual(ws_latency, ring_latency * (1 + tolerance))

    # Test Scenario 2: Heterogeneous Nodes with Uniform Latency
    async def test_heterogeneous_uniform_latency(self):
        print("\nTest Scenario 2: Heterogeneous Nodes with Uniform Latency")
        node_configs = {
            "node1": DeviceCapabilities(
                model="High-End GPU",
                chip="Chip1",
                memory=24000,
                flops=DeviceFlops(fp32=15e12, fp16=30e12, int8=60e12),
                power_usage=300
            ),
            "node2": DeviceCapabilities(
                model="Mid-Range GPU",
                chip="Chip2",
                memory=16000,
                flops=DeviceFlops(fp32=10e12, fp16=20e12, int8=40e12),
                power_usage=200
            ),
            "node3": DeviceCapabilities(
                model="Low-End GPU",
                chip="Chip3",
                memory=8000,
                flops=DeviceFlops(fp32=5e12, fp16=10e12, int8=20e12),
                power_usage=150
            )
        }
        edge_configs = [
            {'from': "node1", 'to': "node2", 'latency': 0.01},
            {'from': "node2", 'to': "node3", 'latency': 0.01},
            {'from': "node1", 'to': "node3", 'latency': 0.01}
        ]
        await self.setup_scenario(node_configs, edge_configs)

        total_layers = 100
        shards = [Shard("model", 0, total_layers - 1, total_layers)]

        ring_total_time, ring_latency = await self.run_simulation(self.ring_strategy, shards)
        ws_total_time, ws_latency = await self.run_simulation(self.work_stealing_strategy, shards)

        print(f"Ring Strategy - Total Time: {ring_total_time:.4f}s, Latency: {ring_latency:.4f}s")
        print(f"Work Stealing Strategy - Total Time: {ws_total_time:.4f}s, Latency: {ws_latency:.4f}s")
        tolerance = 0.01
        self.assertLess(ws_latency, ring_latency * (1 + tolerance))
        self.assertLess(ws_total_time, ring_total_time * (1 + tolerance))

    # Test Scenario 3: Heterogeneous Nodes with Variable Latency
    async def test_heterogeneous_variable_latency(self):
        print("\nTest Scenario 3: Heterogeneous Nodes with Variable Latency")
        node_configs = {
            "node1": DeviceCapabilities(
                model="High-End GPU",
                chip="Chip1",
                memory=24000,
                flops=DeviceFlops(fp32=20e12, fp16=40e12, int8=80e12),
                power_usage=300
            ),
            "node2": DeviceCapabilities(
                model="Mid-Range GPU",
                chip="Chip2",
                memory=16000,
                flops=DeviceFlops(fp32=12e12, fp16=24e12, int8=48e12),
                power_usage=200
            ),
            "node3": DeviceCapabilities(
                model="Low-End GPU",
                chip="Chip3",
                memory=8000,
                flops=DeviceFlops(fp32=6e12, fp16=12e12, int8=24e12),
                power_usage=150
            )
        }
        edge_configs = [
            {'from': "node1", 'to': "node2", 'latency': 0.005},
            {'from': "node2", 'to': "node3", 'latency': 0.02},
            {'from': "node1", 'to': "node3", 'latency': 0.03}
        ]
        await self.setup_scenario(node_configs, edge_configs)

        total_layers = 100
        shards = [Shard("model", 0, total_layers - 1, total_layers)]

        ring_total_time, ring_latency = await self.run_simulation(self.ring_strategy, shards)
        ws_total_time, ws_latency = await self.run_simulation(self.work_stealing_strategy, shards)

        print(f"Ring Strategy - Total Time: {ring_total_time:.4f}s, Latency: {ring_latency:.4f}s")
        print(f"Work Stealing Strategy - Total Time: {ws_total_time:.4f}s, Latency: {ws_latency:.4f}s")

        self.assertLess(ws_latency, ring_latency)
        self.assertLess(ws_total_time, ring_total_time)

    # Test Scenario 4: Node Failure During Execution
    async def test_node_failure(self):
        print("\nTest Scenario 4: Node Failure During Execution")
        node_configs = {
            "node1": DeviceCapabilities(
                model="High-End GPU",
                chip="Chip1",
                memory=24000,
                flops=DeviceFlops(fp32=15e12, fp16=30e12, int8=60e12),
                power_usage=300
            ),
            "node2": DeviceCapabilities(
                model="Mid-Range GPU",
                chip="Chip2",
                memory=16000,
                flops=DeviceFlops(fp32=10e12, fp16=20e12, int8=40e12),
                power_usage=200
            ),
            "node3": DeviceCapabilities(
                model="Low-End GPU",
                chip="Chip3",
                memory=8000,
                flops=DeviceFlops(fp32=5e12, fp16=10e12, int8=20e12),
                power_usage=150
            )
        }
        edge_configs = [
            {'from': "node1", 'to': "node2", 'latency': 0.01},
            {'from': "node2", 'to': "node3", 'latency': 0.01},
            {'from': "node1", 'to': "node3", 'latency': 0.01}
        ]
        await self.setup_scenario(node_configs, edge_configs)

        total_layers = 100
        shards = [Shard("model", 0, total_layers - 1, total_layers)]

        async def simulate_node_failure():
            await asyncio.sleep(0.5)
            self.nodes["node2"].is_active = False
            print("Simulated failure of node2")

        failure_task = asyncio.create_task(simulate_node_failure())
        ring_task = asyncio.create_task(self.run_simulation(self.ring_strategy, shards))
        await asyncio.gather(failure_task, ring_task)
        ring_total_time, ring_latency = ring_task.result()

        # Reset for work stealing strategy
        self.nodes["node2"].is_active = True
        for node in self.nodes.values():
            node.work_done = 0

        failure_task = asyncio.create_task(simulate_node_failure())
        ws_task = asyncio.create_task(self.run_simulation(self.work_stealing_strategy, shards))
        await asyncio.gather(failure_task, ws_task)
        ws_total_time, ws_latency = ws_task.result()

        print(f"Ring Strategy - Total Time: {ring_total_time:.4f}s, Latency: {ring_latency:.4f}s")
        print(f"Work Stealing Strategy - Total Time: {ws_total_time:.4f}s, Latency: {ws_latency:.4f}s")

        self.assertLess(ws_total_time, ring_total_time)

    # Test Scenario 5: High Network Latency Environment
    async def test_high_network_latency(self):
        print("\nTest Scenario 5: High Network Latency Environment")
        node_configs = {
            f"node{i}": DeviceCapabilities(
                model="Generic GPU",
                chip="GenericChip",
                memory=16000,
                flops=DeviceFlops(fp32=10e12, fp16=20e12, int8=40e12),
                power_usage=200
            ) for i in range(1, 4)
        }
        edge_configs = [
            {'from': "node1", 'to': "node2", 'latency': 0.1},
            {'from': "node2", 'to': "node3", 'latency': 0.1},
            {'from': "node1", 'to': "node3", 'latency': 0.1}
        ]
        await self.setup_scenario(node_configs, edge_configs)

        total_layers = 100
        shards = [Shard("model", 0, total_layers - 1, total_layers)]

        ring_total_time, ring_latency = await self.run_simulation(self.ring_strategy, shards)
        ws_total_time, ws_latency = await self.run_simulation(self.work_stealing_strategy, shards)

        print(f"Ring Strategy - Total Time: {ring_total_time:.4f}s, Latency: {ring_latency:.4f}s")
        print(f"Work Stealing Strategy - Total Time: {ws_total_time:.4f}s, Latency: {ws_latency:.4f}s")
        tolerance = 0.01
        self.assertLessEqual(ws_total_time, ring_total_time * (1 + tolerance))
        self.assertLessEqual(ws_latency, ring_latency * (1 + tolerance))

    # Test Scenario 6: High Compute Demand (Large Shards)
    async def test_high_compute_demand(self):
        print("\nTest Scenario 6: High Compute Demand (Large Shards)")
        node_configs = {
            f"node{i}": DeviceCapabilities(
                model="Generic GPU",
                chip="GenericChip",
                memory=16000,
                flops=DeviceFlops(fp32=10e12, fp16=20e12, int8=40e12),
                power_usage=200
            ) for i in range(1, 5)
        }
        edge_configs = [
            {'from': f"node{i}", 'to': f"node{j}", 'latency': 0.01}
            for i in range(1, 5) for j in range(i+1, 5)
        ]
        await self.setup_scenario(node_configs, edge_configs)

        total_layers = 1000  # Large number of layers to simulate high compute demand
        shards = [Shard("model", 0, total_layers - 1, total_layers)]

        ring_total_time, ring_latency = await self.run_simulation(self.ring_strategy, shards)
        ws_total_time, ws_latency = await self.run_simulation(self.work_stealing_strategy, shards)

        print(f"Ring Strategy - Total Time: {ring_total_time:.4f}s, Latency: {ring_latency:.4f}s")
        print(f"Work Stealing Strategy - Total Time: {ws_total_time:.4f}s, Latency: {ws_latency:.4f}s")
        tolerance = 0.01
        self.assertLessEqual(ws_latency, ring_latency * (1 + tolerance))
        self.assertLessEqual(ws_total_time, ring_total_time * (1 + tolerance))

    # Test Scenario 7: Uneven Workload Distribution
    async def test_uneven_workload_distribution(self):
        print("\nTest Scenario 7: Uneven Workload Distribution")
        node_configs = {
            "node1": DeviceCapabilities(
                model="Fast GPU",
                chip="Chip1",
                memory=16000,
                flops=DeviceFlops(fp32=15e12, fp16=30e12, int8=60e12),
                power_usage=250
            ),
            "node2": DeviceCapabilities(
                model="Slow GPU",
                chip="Chip2",
                memory=16000,
                flops=DeviceFlops(fp32=5e12, fp16=10e12, int8=20e12),
                power_usage=150
            )
        }
        edge_configs = [
            {'from': "node1", 'to': "node2", 'latency': 0.01}
        ]
        await self.setup_scenario(node_configs, edge_configs)

        total_layers = 100
        shards = [Shard("model", 0, 49, total_layers), Shard("model", 50, 99, total_layers)]

        ring_total_time, ring_latency = await self.run_simulation(self.ring_strategy, shards)
        ws_total_time, ws_latency = await self.run_simulation(self.work_stealing_strategy, shards)

        print(f"Ring Strategy - Total Time: {ring_total_time:.4f}s, Latency: {ring_latency:.4f}s")
        print(f"Work Stealing Strategy - Total Time: {ws_total_time:.4f}s, Latency: {ws_latency:.4f}s")
        tolerance = 0.01
        self.assertLessEqual(ws_total_time, ring_total_time * (1 + tolerance))
        self.assertLessEqual(ws_latency, ring_latency * (1 + tolerance))

    # Test Scenario 8: Nodes with Varying Power Consumption
    async def test_varying_power_consumption(self):
        print("\nTest Scenario 8: Nodes with Varying Power Consumption")
        node_configs = {
            "node1": DeviceCapabilities(
                model="Efficient GPU",
                chip="Chip1",
                memory=12000,
                flops=DeviceFlops(fp32=12e12, fp16=32e12, int8=48e12),
                power_usage=150
            ),
            "node2": DeviceCapabilities(
                model="Power-Hungry GPU",
                chip="Chip2",
                memory=16000,
                flops=DeviceFlops(fp32=15e12, fp16=24e12, int8=60e12),
                power_usage=300
            )
        }
        edge_configs = [
            {'from': "node1", 'to': "node2", 'latency': 0.01}
        ]
        await self.setup_scenario(node_configs, edge_configs)

        total_layers = 100
        shards = [Shard("model", 0, total_layers - 1, total_layers)]

        # Run simulations and calculate power consumption
        ring_total_time, ring_latency = await self.run_simulation(self.ring_strategy, shards)
        ring_power = sum(node.capabilities.power_usage * ring_total_time for node in self.nodes.values())

        for node in self.nodes.values():
            node.work_done = 0

        ws_total_time, ws_latency = await self.run_simulation(self.work_stealing_strategy, shards)
        ws_power = sum(node.capabilities.power_usage * ws_total_time for node in self.nodes.values())

        print(f"Ring Strategy - Total Time: {ring_total_time:.4f}s, Power: {ring_power:.2f}W")
        print(f"Work Stealing Strategy - Total Time: {ws_total_time:.4f}s, Power: {ws_power:.2f}W")
        tolerance = 0.01
        self.assertLessEqual(ws_power, ring_power * (1 + tolerance))

if __name__ == '__main__':
    unittest.main()
