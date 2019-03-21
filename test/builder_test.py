import sys
sys.path.append("../")
from network_builder import NetworkBuilder
from net_parser import Parser
from network import Network

def main():
    parser = Parser('../data/alexnet.cfg')
    network_builder = NetworkBuilder("test")
    network = network_builder.set_parser(parser).build() # type: Network
        

if __name__ == '__main__':
    main()