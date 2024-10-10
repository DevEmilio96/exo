import asyncio
import time
from typing import List, Dict, Tuple
from exo.networking.udp.udp_discovery import UDPDiscovery
from exo.networking.peer_handle import PeerHandle
from exo.topology.topology import Topology
from exo.topology.device_capabilities import DeviceCapabilities
import logging

logger = logging.getLogger(__name__)

class MetricsUDPDiscovery(UDPDiscovery):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.topology = Topology()
        self.topology.update_node(self.node_id, self.device_capabilities)
        self.network_metrics: Dict[Tuple[str, str], Dict[str, float]] = {}
        self.last_metric_update: Dict[str, float] = {}
        self.metric_update_interval = 60  # Seconds
        logger.info(f"MetricsUDPDiscovery initialized with node_id: {self.node_id}")

    async def start(self):
        await super().start()
        self.metric_update_task = asyncio.create_task(self.periodic_metric_update())

    async def stop(self):
        if hasattr(self, 'metric_update_task'):
            self.metric_update_task.cancel()
            try:
                await self.metric_update_task
            except asyncio.CancelledError:
                pass
        await super().stop()

    async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
        peers = await super().discover_peers(wait_for_peers)
        if peers:
            logger.info(f"Discovered {len(peers)} peers: {[peer.id() for peer in peers]}")
            await self.update_network_metrics(peers)
        else:
            logger.warning("No peers discovered")
        return peers

    async def update_network_metrics(self, peers: List[PeerHandle]):
        update_tasks = [self.measure_network_metrics(peer) for peer in peers]
        await asyncio.gather(*update_tasks, return_exceptions=True)
        logger.info(f"Updated topology: {self.topology}")

    async def measure_network_metrics(self, peer: PeerHandle):
        try:
            current_time = time.time()
            if current_time - self.last_metric_update.get(peer.id(), 0) < self.metric_update_interval:
                return  # Skip if updated recently

            logger.info(f"Measuring network metrics for peer: {peer.id()}")
            latency = await self.measure_latency(peer)
            self.network_metrics[(self.node_id, peer.id())] = {"latency": latency}
            self.topology.update_edge(self.node_id, peer.id(), latency=latency)
            self.last_metric_update[peer.id()] = current_time
            logger.info(f"Network metrics for {self.node_id} -> {peer.id()}: Latency={latency}")
        except Exception as e:
            logger.error(f"Error measuring metrics for peer {peer.id()}: {e}")

    async def measure_latency(self, peer: PeerHandle) -> float:
        start_time = time.time()
        try:
            await asyncio.wait_for(peer.health_check(), timeout=5.0)
            latency = time.time() - start_time
            logger.info(f"Measured latency for peer {peer.id()}: {latency}")
            return latency
        except asyncio.TimeoutError:
            logger.warning(f"Latency measurement timed out for peer {peer.id()}")
            return float('inf')
        except Exception as e:
            logger.error(f"Error during latency measurement for peer {peer.id()}: {e}")
            return float('inf')

    async def on_listen_message(self, data, addr):
        await super().on_listen_message(data, addr)
        # We'll update metrics periodically instead of on every message

    async def task_cleanup_peers(self):
        while True:
            try:
                await super().task_cleanup_peers()
                active_peer_ids = set(self.known_peers.keys()) | {self.node_id}
                for node_id in list(self.topology.nodes.keys()):
                    if node_id not in active_peer_ids:
                        self.topology.nodes.pop(node_id, None)
                        self.topology.peer_graph.pop(node_id, None)
                        self.last_metric_update.pop(node_id, None)
                if self.node_id not in self.topology.nodes:
                    self.topology.update_node(self.node_id, self.device_capabilities)
                logger.info(f"Updated topology after cleanup: {self.topology}")
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
            finally:
                await asyncio.sleep(self.broadcast_interval)

    async def periodic_metric_update(self):
        while True:
            try:
                peers = [peer for peer, _, _, _ in self.known_peers.values()]
                await self.update_network_metrics(peers)
            except Exception as e:
                logger.error(f"Error in periodic metric update: {e}")
            await asyncio.sleep(self.metric_update_interval)

    def get_topology(self) -> Topology:
        if not self.topology.nodes:
            logger.warning("Topology is empty. Adding current node.")
            self.topology.update_node(self.node_id, self.device_capabilities)
        logger.info(f"Getting topology: {self.topology}")
        return self.topology