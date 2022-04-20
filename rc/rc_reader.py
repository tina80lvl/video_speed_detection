
import numpy as np
import json


def read_rc(fname):
    file_n = 0
    data = np.fromfile(fname, dtype=np.byte)
    # print(len(data))
    idx = 0

    frame_w = 0
    frame_h = 0

    summ = 0

    licnums_by_frame = list()
    while idx < len(data):
        magic = np.frombuffer(data[idx: idx+4], dtype=np.uint32)[0]
        print('magic:', magic)
        idx += 4

        frame_len = np.frombuffer(data[idx: idx+4], dtype=np.uint32)[0]
        summ += frame_len
        # print('frame_len:', frame_len)
        idx += 4

        sub_data = data[idx: idx+frame_len-8]

        sub_idx = 0

        frame_info = dict()
        while sub_idx < frame_len - 8:
            type = np.frombuffer(sub_data[sub_idx: sub_idx+2], dtype=np.uint16)[0]

            # print('type:', type)

            sub_idx += 2
            sub_len = np.frombuffer(sub_data[sub_idx: sub_idx+4], dtype=np.uint32)[0]
            # print(sub_len)
            sub_idx += 4

            if type == 21:
                frame_size = np.frombuffer(sub_data[sub_idx: sub_idx+4], dtype=np.uint32)[0]

            if type == 3:
                frame_id = np.frombuffer(sub_data[sub_idx: sub_idx+4], dtype=np.uint32)[0]

            if type == 0:
                seconds = np.frombuffer(sub_data[sub_idx: sub_idx+4], dtype=np.uint32)[0]
                microseconds = np.frombuffer(sub_data[sub_idx + 4: sub_idx + 8], dtype=np.uint32)[0]
                frame_info['seconds'] = seconds
                frame_info['microseconds'] = microseconds

            if type == 5:
                info = np.frombuffer(sub_data[sub_idx: sub_idx+sub_len], dtype=np.byte)
                num_area = len(info) / 92
                area_list = np.array_split(info, num_area)

                # print('file-number:', file_n)
                # print(identifiacator)
                licnums = []
                for area in area_list:
                    licnum = dict()
                    # сколько символов будет в тексте
                    n_symbols = np.frombuffer(area[4:8], dtype=np.int32)[0]
                    # print('n_symbols:', n_symbols)

                    # сам текст с определенным количетсвом символов
                    text = np.frombuffer(area[8:40], dtype=np.uint16)
                    print('text:', str(bytes(text[0:n_symbols]).decode('utf-16')))
                    licnum['text'] = str(bytes(text[0:n_symbols]).decode('utf-16'))
                    # x координаты
                    x_coord = area[74:82]
                    x_coord_list = list(map(lambda x: np.frombuffer(x, dtype=np.uint16)[0], np.array_split(x_coord, 4)))
                    # print('x coordinates:', x_coord_list)
                    licnum['x'] = x_coord_list

                    # y координаты
                    y_coord = area[82:90]
                    y_coord_list = list(map(lambda y: np.frombuffer(y, dtype=np.uint16)[0], np.array_split(y_coord, 4)))
                    # print('y coordinates:', y_coord_list)
                    licnum['y'] = y_coord_list

                    licnums.append(licnum)
                frame_info['licnums'] = licnums

            if type == 19:  # Ширина картинки.
                frame_w = np.frombuffer(sub_data[sub_idx: sub_idx+4], dtype=np.uint32)[0]
                # print('frame_w:', frame_w)

            if type == 20:  # Высота картинкт.
                frame_h = np.frombuffer(sub_data[sub_idx: sub_idx+4], dtype=np.uint32)[0]
                # print(frame_h)
                # print(sub_len)

            if type == 22:  # Формат картинки.
                img_type = sub_data[sub_idx: sub_idx+sub_len]
                # print(sub_len)
                # print('image_type:', repr(bytes(img_type).decode('utf-8')))

            if type == 23 and frame_w > 0 and frame_h > 0:  # Сама картинка.
                img_data = sub_data[sub_idx: sub_idx+sub_len]

                file_n += 1
                # print('file_n:', file_n)
                # print(sub_len)

                # Записываем изображение в файл
                # img_data.tofile("test/" + f'{file_n:06d}' + ".jpg")

            sub_idx += sub_len
        # print('frame', frame_info)
        licnums_by_frame.append(frame_info)
        idx += frame_len-8
    # print(summ)
    return licnums_by_frame


fname = '/Users/tina/dev/simicon/rc/out/out.rc'
licnums = read_rc(fname)
print(licnums)

dict = [{'seconds': 1637925286, 'microseconds': 79683, 'licnums': [{'text': 'х026сх178', 'x': [1766, 1700, 1700, 1766], 'y': [823, 809, 823, 809]}]}]

# json = json.dumps(dict)
# print(json)
with open('licnums.json', 'w') as f:
    f.write(str(licnums))
    # json.dump(dict, f, indent = 4)
