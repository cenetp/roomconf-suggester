import random
import os


def check_room(existing_rooms, room_type, abstraction):
    is_valid_step = False
    for k in range(len(existing_rooms)):
        room_type_prev = existing_rooms[k][1]
        abstraction_prev = existing_rooms[k][0]
        if abstraction != '':
            if room_type == room_type_prev and abstraction == abstraction_prev:
                is_valid_step = True
                break
        else:
            if room_type == room_type_prev:
                is_valid_step = True
                break
    return is_valid_step


def room_exists(existing_rooms, room_type, action, abstraction, pos):
    is_valid_step = False
    if len(existing_rooms) > 0 and action != 'a':
        is_valid_step = check_room(existing_rooms, room_type, abstraction)
    elif action == 'a':  # if step is 'add'
        if pos != '':
            is_valid_step = check_room(existing_rooms, room_type, abstraction)
        else:
            is_valid_step = True
    return is_valid_step


def generate_datastring(num_steps):
    actions = [
        'a',  # add - the neighbouring rooms should exist to prevent isolated rooms
        'r',  # remove - the room should exist, it is implicated that the connections of this room should be removed too
        't'  # change type - the room should exist
        #  'f',  # reshape, change form - the room should exist
    ]

    abstractions = [
        '0'  # abstract, without shape
        #  '1'  # complex, with shape
    ]

    positions = [
        'b',  # between, both rooms should be connected to each other
        'n'  # next to
    ]

    room_types = [
        # 'R',  # room
        'L',  # living
        'S',  # sleeping
        'W',  # working
        'K',  # kitchen
        'C',  # corridor
        'B',  # bath
        'T',  # toilet
        'H',  # children
        'G'  # storage
    ]

    edge_types = [
        'E',  # edge
        'D',  # door
        'P',  # passage
        'W',  # wall
        'R',  # ENTRANCE
        'B',  # SLAB
        'S',  # STAIRS
        'N'  # WINDOW
    ]

    # Some possible steps
    # N. <mandatory group>-[<position group>-<connection group>]
    # 1. a1L-bWS-DD
    # 2. a0S-nC
    # 3. r1S-bKW
    # 4. c1WL

    f = open(os.getcwd() + '/tf_scripts/suggester/process.csv', 'a')
    f.write('ID,Step,RoomCount,EdgeCount,FPcount,SubStepsCount,ExistingRooms\n')

    for i in range(num_steps):
        initial_abstraction = abstractions[random.randrange(0, len(abstractions))]
        initial_room_type = room_types[random.randrange(0, len(room_types))]
        step = 'a' + initial_abstraction + initial_room_type + ':;'
        # e.g.: a1L = add non-abstract living room
        existing_edges = []
        existing_rooms = [initial_abstraction + initial_room_type]
        j = 0
        num_sub_steps = random.randrange(10, 60)
        while j < num_sub_steps:  # number of sub-steps
            action = actions[random.randrange(0, len(actions))]
            abstraction = abstractions[random.randrange(0, len(abstractions))]
            room_type = room_types[random.randrange(0, len(room_types))]
            # e.g.: r0K
            ar = abstraction + room_type
            is_valid_step = room_exists(existing_rooms, room_type, action, abstraction, '')

            if is_valid_step is True:
                other_groups = ''
                if action == 'a':
                    position = positions[random.randrange(0, len(positions))]
                    if position == 'b' and len(existing_rooms) >= 2:
                        room_type1 = ''
                        room_type2 = ''
                        while True:
                            room_type1 = room_types[random.randrange(0, len(room_types))]
                            room1_exists = room_exists(existing_rooms, room_type1, action, '', position)
                            existing_rooms_temp = []
                            if room1_exists:
                                for r in existing_rooms:
                                    existing_rooms_temp.append(r)
                                try:
                                    existing_rooms_temp.remove('0' + room_type1)
                                except:
                                    existing_rooms_temp.remove('1' + room_type1)
                                room_type2 = room_types[random.randrange(0, len(room_types))]
                                room2_exists = room_exists(existing_rooms_temp, room_type2, action, '', position)
                            else:
                                room2_exists = False
                            if room1_exists is True and room2_exists is True:
                                break
                        other_groups += room_type1 + room_type2
                        edge_type1 = edge_types[random.randrange(0, len(edge_types))]
                        edge_type2 = edge_types[random.randrange(0, len(edge_types))]
                        existing_edges.append(edge_type1)
                        existing_edges.append(edge_type2)
                        other_groups += '-' + edge_type1 + edge_type2
                    elif len(existing_rooms) > 1:
                        room_type1 = ''
                        while True:
                            room_type1 = room_types[random.randrange(0, len(room_types))]
                            room1_exists = room_exists(existing_rooms, room_type1, action, '', position)
                            if room1_exists is True:
                                break
                        other_groups += room_type1
                        if action == 'a':
                            edge_type1 = edge_types[random.randrange(0, len(edge_types))]
                            existing_edges.append(edge_type1)
                            other_groups += '-' + edge_type1
                    if other_groups != '':
                        other_groups = position + other_groups
                    existing_rooms.append(ar)
                elif action == 't' and len(existing_rooms) >= 1:
                    while True:
                        room_type1 = room_types[random.randrange(0, len(room_types))]
                        if room_type1 != ar[1]:
                            break
                    existing_rooms.remove(ar)
                    existing_rooms.append(ar[0] + room_type1)
                    other_groups += '(' + ar[0] + room_type1 + ')'
                elif action == 'r' and len(existing_rooms) >= 1:
                    existing_rooms.remove(ar)
                step += action + abstraction + room_type + ':' + other_groups + ';'

                j += 1

        if step[len(step) - 1] == ';':
            step = step[:-1]
            if step[len(step) - 1] == ';':
                step = step[:-1]

        room_count = len(existing_rooms)
        edge_count = len(existing_edges)
        add_fp = random.choice([True, False])
        fp_count_str = ''
        if add_fp is True:
            fp_count_str += str(random.randrange(1, 8))
        else:
            fp_count_str += '0'

        step = str(i) + ',' + step + ',' + str(room_count) + ',' + \
               str(edge_count) + ',' + fp_count_str + ',' + str(num_sub_steps) + ',' + '_'.join(existing_rooms) + '\n'
        # print(step)
        f.write(step)

    print('Chain data generated.')


generate_datastring(250000)
