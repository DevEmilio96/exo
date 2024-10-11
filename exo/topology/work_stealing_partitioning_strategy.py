import asyncio
from typing import List, Dict, Optional, Set
from exo.topology.partitioning_strategy import PartitioningStrategy, Partition
from exo.topology.topology import Topology
from exo.inference.shard import Shard
import logging

logger = logging.getLogger(__name__)

class WorkStealingPartitioningStrategy(PartitioningStrategy):
    def __init__(self):
        self.work_queue: Dict[str, List[Shard]] = {}
        self.node_capabilities: Dict[str, float] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        self.active_nodes: Set[str] = set()
        self.topology: Optional[Topology] = None

    def partition(self, topology: Topology) -> List[Partition]:
        self.topology = topology
        nodes = list(topology.nodes.keys())
        node_capabilities = [self._calculate_node_capability(node_id) for node_id in nodes]
        total_capability = sum(node_capabilities)

        partitions = []
        cumulative = 0.0

        if total_capability <= 0:
            logger.warning("Total capability is zero or negative. Using equal partitions.")
            partition_size = 1.0 / len(nodes)
            for i, node_id in enumerate(nodes):
                partitions.append(Partition(node_id=node_id, start=i*partition_size, end=(i+1)*partition_size))
                self._initialize_node(node_id, partition_size)
        else:
            for node_id, capability in zip(nodes, node_capabilities):
                node_partition = capability / total_capability
                partitions.append(Partition(node_id=node_id, start=cumulative, end=cumulative + node_partition))
                self._initialize_node(node_id, node_partition)
                cumulative += node_partition

        logger.debug(f"Partitions: {partitions}")
        return partitions

    def _initialize_node(self, node_id: str, capability: float):
        self.work_queue[node_id] = []
        self.locks[node_id] = asyncio.Lock()
        self.node_capabilities[node_id] = capability
        self.active_nodes.add(node_id)

    async def get_work(self, node_id: str) -> Optional[Shard]:
        async with self.locks[node_id]:
            my_work_count = len(self.work_queue.get(node_id, []))
            if my_work_count <= 2:
                node_work_counts = {nid: len(self.work_queue.get(nid, [])) for nid in self.active_nodes}
                potential_donors = [
                    nid for nid in self.active_nodes
                    if nid != node_id and node_work_counts.get(nid, 0) > my_work_count
                ]

                potential_donors.sort(key=lambda nid: -node_work_counts.get(nid, 0))

                for donor_node_id in potential_donors:
                    async with self.locks[donor_node_id]:
                        donor_work_count = len(self.work_queue.get(donor_node_id, []))
                        if donor_work_count > my_work_count:
                            stolen_shard = self.work_queue[donor_node_id].pop(0)
                            logger.info(f"Node {node_id} stole work from {donor_node_id}")
                            logger.debug(f"Work distribution: {node_work_counts}")
                            return stolen_shard

            if self.work_queue.get(node_id):
                return self.work_queue[node_id].pop(0)

        return None

    async def add_work(self, node_id: str, shard: Shard):
        if node_id in self.active_nodes:
            async with self.locks[node_id]:
                self.work_queue[node_id].append(shard)

    def is_work_remaining(self) -> bool:
        return any(
            self.work_queue.get(nid) and self.topology.nodes[nid].is_active
            for nid in self.active_nodes
        )

    def _calculate_node_capability(self, node_id: str) -> float:
        node = self.topology.nodes[node_id]
        capability = node.flops.fp16
        logger.debug(f"Node {node_id} capability: {capability:.5f} (FLOPS: {capability:.2f})")
        return capability