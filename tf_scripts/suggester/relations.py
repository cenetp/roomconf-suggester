import operator
import random
from argparse import ArgumentParser
from collections import Counter

parser = ArgumentParser()
parser.add_argument('-e', '--edges', dest='edges_str', required=True)
parser.add_argument('-n', '--number', dest='number', required=True)
parser.add_argument('-r', '--room', dest='room', required=True)
parser.add_argument('-c', '--conntype', dest='conn_type', required=True)
args = parser.parse_args()


def create_mapping(edges_str):
    rooms_edges = str(edges_str).strip('\'').split(',')
    rooms = []
    edges = []
    rooms_edges_combined = {}
    for i in range(len(rooms_edges)):
        room_edges = rooms_edges[i].split('-')
        rooms.append(room_edges[0])
        edges.append(room_edges[1])
    for j in range(len(rooms)):
        try:
            rooms_edges_combined[rooms[j]]
        except KeyError:
            rooms_edges_combined[rooms[j]] = []
        rooms_edges_combined[rooms[j]].append(edges[j])
    for room in rooms_edges_combined:
        edges_list = rooms_edges_combined[room]
        new_edges_list = []
        for edges in edges_list:
            if len(edges) > 2:
                multiple_edges = []
                for k in range(len(edges)):
                    e = edges[k]
                    for l in range(len(edges)):
                        if k != l:
                            multiple_edges.append(str(e + edges[l]))
                new_edges_list.extend(multiple_edges)
            else:
                new_edges_list.append(edges)
        conn_count = dict(Counter(new_edges_list))
        rooms_edges_combined[room] = conn_count
    for room in rooms_edges_combined:
        room_conn = []
        for conn in rooms_edges_combined[room]:
            conn_list = [conn, str(rooms_edges_combined[room][conn])]
            room_conn.append(tuple(conn_list))
        room_conn.sort(key=operator.itemgetter(1), reverse=True)
        rooms_edges_combined[room] = room_conn
    return rooms_edges_combined


def suggest_relation(room, num, conn_type):
    mapping = create_mapping(args.edges_str)
    num = int(num)
    if room in mapping:
        relation = ''
        while num < len(mapping[room]):
            rel, count = mapping[room][num]
            if len(rel) == 2 and conn_type == 'b':
                relation = rel
                break
            elif len(rel) == 1 and conn_type == 'n':
                relation = rel
                break
            else:
                num += 1
        return relation
    else:
        # TODO import from generate_data
        edge_types = [
            # 'E',  # edge
            'D',  # door
            'P',  # passage
            'W',  # wall
            'R',  # ENTRANCE
            'B',  # SLAB
            'S',  # STAIRS
            'N'  # WINDOW
        ]
        if conn_type == 'b':
            edge_type1 = edge_types[random.randrange(0, len(edge_types))]
            edge_type2 = edge_types[random.randrange(0, len(edge_types))]
            return edge_type1+edge_type2
        elif conn_type == 'n':
            edge_type1 = edge_types[random.randrange(0, len(edge_types))]
            return edge_type1


print(suggest_relation(args.room, args.number, args.conn_type))
