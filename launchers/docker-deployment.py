#launchers/docker-deployment.py

"""Provide deployment in docker.

This module allows the user to start the federated learning in production mode with docker.

The launcher can be run with the command

`python launchers/docker-deployment.py`

Attributes:
    --n_clients (int): The number of clients to run .Can be abbreviate with -nc. Defaults to 2
    --build (str): Image(s) to build the images, "client" will build only client app, "server" only the server app and "all" both images. Can be abbreviate with -b. Defaults to ""
    --keep (boolean): Flag to indicate if the script should remove the container and network after run or not. Can be abbreviate with -k. Defaults to False. 
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
    parser.add_argument('-nc', '--n_clients', type=int, default=2, help="Number of clients in the federation")
    parser.add_argument('-b', "--build", type=str, default="", help='Build images "client" for client image, "server" for server image "all" for both')

    # Flags
    parser.add_argument('-k', "--keep", action="store_true", help="Keep docker containers and network after run")

    return parser.parse_args()


def build_client_image() -> None:
    """Remove oldest client image and build new one."""
    subprocess.run(['sudo', 'docker', 'image', 'rm', 'flwr_clientapp:0.0.1'])
    subprocess.run(['sudo', 'docker', 'build', '-f', 'clientapp.Dockerfile', '-t', 'flwr_clientapp:0.0.1', '.'])


def build_server_image():
    """Remove oldest server image and build new one."""
    subprocess.run(['sudo', 'docker', 'image', 'rm', 'flwr_serverapp:0.0.1'])
    subprocess.run(['sudo', 'docker', 'build', '-f', 'serverapp.Dockerfile', '-t', 'flwr_serverapp:0.0.1', '.'])


def build_images(req: str) -> None:
    """Build server and client images
    
    Args:
        req: The given instructions for builds (`client`, `server`, `all`)
    """
    match req:
        case "client":
            build_client_image()

        case "server":
            build_server_image()
        
        case "all":
            build_client_image()
            build_server_image()
        
        case _:
            raise KeyError(f'build flag should be "client", "server" or "all" and not {req}')   


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
            '--node-config', f'partition-id={i} num-partitions={n_clients}'
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
    if opt.build != "":
        build_images(opt.build)

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
    run_clients(opt.n_clients)

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
        clean(opt.n_clients)