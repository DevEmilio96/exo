import asyncio
import time
from typing import List, Tuple

import numpy as np
from exo.inference.shard import Shard
from exo.networking.udp.udp_discovery import UDPDiscovery
from exo.networking.peer_handle import PeerHandle
from exo.topology.topology import Topology
from exo.topology.device_capabilities import DeviceCapabilities
from exo import DEBUG, DEBUG_DISCOVERY

import logging
logger = logging.getLogger(__name__)

class MetricsUDPDiscovery(UDPDiscovery):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.topology = Topology()
        logger.info(f"MetricsUDPDiscovery initialized with node_id: {self.node_id}")

    async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
        peers = await super().discover_peers(wait_for_peers)
        logger.info(f"Discovered {len(peers)} peers: {[peer.id() for peer in peers]}")
        await self.update_network_metrics(peers)
        return peers

    async def update_network_metrics(self, peers: List[PeerHandle]):
        logger.info("Updating network metrics")
        update_tasks = [self.measure_network_metrics(peer) for peer in peers]
        await asyncio.gather(*update_tasks)
        logger.info(f"Updated topology: {self.topology}")

    async def measure_network_metrics(self, peer: PeerHandle):
        logger.info(f"Measuring network metrics for peer: {peer.id()}")
        latency = await self.measure_latency(peer)
        throughput = await self.measure_throughput(peer)
        self.topology.update_edge(self.node_id, peer.id(), latency=latency, throughput=throughput)
        logger.info(f"Network metrics for {self.node_id} -> {peer.id()}: Latency={latency}, Throughput={throughput}")

    async def measure_latency(self, peer: PeerHandle) -> float:
        start_time = time.time()
        await peer.health_check()
        end_time = time.time()
        latency = end_time - start_time
        logger.info(f"Measured latency for peer {peer.id()}: {latency}")
        return latency

    async def measure_throughput(self, peer: PeerHandle) -> float:
        data_size = 1024 * 1024  # 1 MB
        data = np.random.bytes(data_size)
        start_time = time.time()
        await peer.send_tensor(Shard(model_id="test", start_layer=0, end_layer=1, n_layers=1), 
                               np.frombuffer(data, dtype=np.uint8))
        end_time = time.time()
        throughput = data_size / (end_time - start_time)
        logger.info(f"Measured throughput for peer {peer.id()}: {throughput}")
        return throughput

    async def on_listen_message(self, data, addr):
        await super().on_listen_message(data, addr)
        for peer_id, (peer_handle, _, _, _) in self.known_peers.items():
            await self.measure_network_metrics(peer_handle)

    async def task_cleanup_peers(self):
        while True:
            await super().task_cleanup_peers()
            active_peer_ids = set(self.known_peers.keys())
            for node_id in list(self.topology.nodes.keys()):
                if node_id not in active_peer_ids and node_id != self.node_id:
                    self.topology.nodes.pop(node_id, None)
                    self.topology.peer_graph.pop(node_id, None)
            logger.info(f"Updated topology after cleanup: {self.topology}")
            await asyncio.sleep(self.broadcast_interval)

    def get_topology(self) -> Topology:
        logger.info(f"Getting topology: {self.topology}")
        return self.topology

    def update_node(self, node_id: str, device_capabilities: DeviceCapabilities):
        self.topology.update_node(node_id, device_capabilities)
        if DEBUG >= 2:
            logger.info(f"Updated node {node_id} in topology with capabilities: {device_capabilities}")