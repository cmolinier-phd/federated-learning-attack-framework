#launchers/docker-deployment.py

"""Provide deployment in docker.

This module allows the user to start the federated learning in production mode with docker.
"""
import argparse
import subprocess


def parse_args() -> argparse.Namespace:
    """Parse run arguments
    
    Returns:
        a namespace with parameters and values associations
    """
    parser = argparse.ArgumentParser()
    
    # Parameters
    parser.add_argument('-nc', '--num_clients', type=int, default=2, help="Number of clients in the federation")

    # Flags
    parser.add_argument('-b', "--build", action="store_true", help="Build images")
    parser.add_argument('-k', "--keep", action="store_true", help="Keep docker containers and netwrok after run")

    return parser.parse_args()


def build_images() -> None:
    """Build server and client images"""
    # Remove old images
    subprocess.run(['sudo', 'docker', 'image', 'rm', 'flwr_serverapp:0.0.1'])
    subprocess.run(['sudo', 'docker', 'image', 'rm', 'flwr_clientapp:0.0.1'])
    
    # Build new images
    subprocess.run(['sudo', 'docker', 'build', '-f', 'serverapp.Dockerfile', '-t', 'flwr_serverapp:0.0.1', '.'])
    subprocess.run(['sudo', 'docker', 'build', '-f', 'clientapp.Dockerfile', '-t', 'flwr_clientapp:0.0.1', '.'])


def run_clients(n_clients: int) -> None:
    """Run client app containers

    Args:
        n_clients: number of clients to launch 
    """

    for i in range(n_clients):
        port = 9094+i

        subprocess.run([
            'sudo', 'docker', 'run', '--rm',
            '-p', f'{port}:{port}',
            '--network', 'flwr-network',
            '--name', f'supernode-{i}',
            '--detach',
            'flwr/supernode:1.13.1',
            '--insecure',
            '--superlink', 'superlink:9092',
            '--clientappio-api-address', f'0.0.0.0:{port}',
            '--isolation', 'process',
            '--node-config', f'id={i} num-partitions={n_clients}'
        ])

        subprocess.run([
            'sudo', 'docker', 'run', '--rm',
            '--network', 'flwr-network',
            '--name', f'clientapp-{i}',
            '--detach',
            '--gpus', 'all',
            'flwr_clientapp:0.0.1',
            '--insecure',
            '--clientappio-api-address', f'supernode-{i}:{port}'
        ])


def clean(n_clients: int) -> None:
    """Remove all created containers and networks
    
    Args:
        n_clients: number of launched clients
    """

    # stop serverapp and superlink
    subprocess.run(['sudo', 'docker', 'stop', 'serverapp', 'superlink'])

    # Stop supernodes and clients
    for i in range(n_clients):
        subprocess.run(['sudo', 'docker', 'stop', f'supernode-{i}'])
        subprocess.run(['sudo', 'docker', 'stop', f'clientapp-{i}'])

    # Remove Flower network
    subprocess.run(['sudo', 'docker', 'network', 'rm', 'flwr-network'])

    # Prune containers
    subprocess.run(['sudo', 'docker', 'container', 'prune', '-f'])


####################################################################################################
#                                                MAIN                                              #
####################################################################################################
if __name__ == "__main__":
    opt = parse_args()
    opt
    
    # Create network
    subprocess.run(['sudo', 'docker', 'network', 'create', '--driver', 'bridge', 'flwr-network'])

    # Build if required
    if opt.build :
        build_images()

    # Run supernode
    subprocess.run([
        'sudo', 'docker', 'run',
        '-p' '9091:9091', '-p', '9092:9092', '-p', '9093:9093',
        '--network', 'flwr-network',
        '--name', 'superlink',
        '--detach',
        'flwr/superlink:1.13.1',
        '--insecure',
        '--isolation', 'process'
    ])
    
    # Launch clients
    run_clients(opt.num_clients)

    # Launch server
    subprocess.run([
        'sudo', 'docker', 'run',
        '--network', 'flwr-network',
        '--name', 'serverapp',
        '--detach',
        '--gpus', 'all',
        'flwr_serverapp:0.0.1',
        '--insecure',
        '--serverappio-api-address', 'superlink:9091'
    ])

    # Run FL process
    subprocess.run(['flwr', 'run', '.', 'local-deployment', '--stream'])

    # Cleaning
    if not opt.keep :
        clean(opt.num_clients)