import numpy as np
import json
# from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename


def read_rc(fname):
    file_n = 0
    data = np.fromfile(fname, dtype=np.byte)
    # print(len(data))
    idx = 0

    frame_w = 0
    frame_h = 0

    summ = 0

    licnums_by_frame = dict()
    licnums_by_frame['frames'] = list()
    while idx < len(data):
        # magic = np.frombuffer(data[idx: idx+4], dtype=np.uint16)[0]
        magic = np.frombuffer(data[idx: idx + 4], dtype=np.uint32)[0]
        # print('magic:', magic)
        idx += 4

        # frame_len = np.frombuffer(data[idx: idx+4], dtype=np.uint16)[0]
        frame_len = np.frombuffer(data[idx: idx + 4], dtype=np.uint32)[0]
        summ += frame_len
        # print('frame_len:', frame_len)
        idx += 4

        sub_data = data[idx: idx + frame_len - 8]

        sub_idx = 0

        frame_info = dict()
        while sub_idx < frame_len - 8:
            type = \
            np.frombuffer(sub_data[sub_idx: sub_idx + 2], dtype=np.uint16)[0]

            # print('type:', type)

            sub_idx += 2
            # sub_len = np.frombuffer(sub_data[sub_idx: sub_idx+4], dtype=np.uint16)[0]
            sub_len = \
            np.frombuffer(sub_data[sub_idx: sub_idx + 4], dtype=np.uint32)[0]
            # print(sub_len)
            sub_idx += 4

            if type == 0:
                # seconds = np.frombuffer(sub_data[sub_idx: sub_idx+4], dtype=np.uint16)[0]
                seconds = \
                np.frombuffer(sub_data[sub_idx: sub_idx + 4], dtype=np.uint32)[
                    0]
                # microseconds = np.frombuffer(sub_data[sub_idx + 4: sub_idx + 8], dtype=np.uint16)[0]
                microseconds = np.frombuffer(sub_data[sub_idx + 4: sub_idx + 8],
                                             dtype=np.uint32)[0]
                frame_info['seconds'] = int(seconds)
                frame_info['microseconds'] = int(microseconds)

            if type == 3:
                # frame_id = np.frombuffer(sub_data[sub_idx: sub_idx+4], dtype=np.uint16)[0]
                frame_id = \
                np.frombuffer(sub_data[sub_idx: sub_idx + 4], dtype=np.uint32)[
                    0]

            if type == 5:  # Информация о ГРЗ
                info = np.frombuffer(sub_data[sub_idx: sub_idx + sub_len],
                                     dtype=np.byte)
                num_area = len(info) / 92
                area_list = np.array_split(info, num_area)

                # print('file-number:', file_n)
                # print(identifiacator)
                licnums = []
                for area in area_list:
                    licnum = dict()
                    # Код страны + номер внутри страны
                    num_format = np.frombuffer(area[:4], dtype=np.int32)[0]
                    licnum['format'] = int(num_format)

                    # сколько символов будет в тексте
                    n_symbols = np.frombuffer(area[4:8], dtype=np.int32)[0]
                    # print('n_symbols:', n_symbols)

                    # сам текст с определенным количетсвом символов
                    text = np.frombuffer(area[8:40], dtype=np.uint16)
                    licnum['text'] = str(
                        bytes(text[0:n_symbols]).decode('utf-16'))
                    # x координаты
                    x_coord = area[74:82]
                    x_coord_list = list(
                        map(lambda x: np.frombuffer(x, dtype=np.uint16)[0],
                            np.array_split(x_coord, 4)))
                    # print('x coordinates:', x_coord_list)
                    licnum['x'] = list(map(int, x_coord_list))

                    # y координаты
                    y_coord = area[82:90]
                    y_coord_list = list(
                        map(lambda y: np.frombuffer(y, dtype=np.uint16)[0],
                            np.array_split(y_coord, 4)))
                    # print('y coordinates:', y_coord_list)
                    licnum['y'] = list(map(int, y_coord_list))

                    licnums.append(licnum)
                frame_info['licnums'] = licnums

            radar_targets = []
            if type == 8:  # Радарные данные.
                '''
                    int id; 4
                    double x; 8
                    double y; 8
                    double xspeed; 8
                    double yspeed; 8
                    double len; 8
                
                    // unused values, only for sizeof(umrr_target) not changed
                    double imgx_unused; 8
                    double imgy_unused; 8
                    int numw_unused; 4
                
                    double imgx_left_unused; 8
                    double imgx_right_unused; 8
                
                    double imgy_top_unused; 8
                    double imgy_bottom_unused; 8
                '''
                info = np.frombuffer(sub_data[sub_idx: sub_idx + sub_len],
                                     dtype=np.byte)
                num_area = len(info) / (11 * 8 + 2 * 4)
                area_list = np.array_split(info, num_area)
                for area in area_list:
                    radar_target = dict()
                    id = np.frombuffer(area[:4], dtype=np.int32)[0]
                    radar_target['id'] = int(id)
                    x = np.frombuffer(area[4:12], dtype=np.double)[0]
                    radar_target['x'] = x
                    y = np.frombuffer(area[12:20], dtype=np.double)[0]
                    radar_target['y'] = y
                    xspeed = np.frombuffer(area[20:28], dtype=np.double)[0]
                    radar_target['xspeed'] = xspeed
                    yspeed = np.frombuffer(area[28:36], dtype=np.double)[0]
                    radar_target['yspeed'] = yspeed
                    carlen = np.frombuffer(area[36:44], dtype=np.double)[0]
                    radar_target['carlen'] = carlen

                    # skipping others
                    radar_targets.append(radar_target)
                frame_info['radar_targets'] = radar_targets

            if type == 19:  # Ширина картинки.
                # frame_w = np.frombuffer(sub_data[sub_idx: sub_idx+4], dtype=np.uint16)[0]
                frame_w = \
                np.frombuffer(sub_data[sub_idx: sub_idx + 4], dtype=np.uint32)[
                    0]
                # print('frame_w:', frame_w)

            if type == 20:  # Высота картинкт.
                # frame_h = np.frombuffer(sub_data[sub_idx: sub_idx+4], dtype=np.uint16)[0]
                frame_h = \
                np.frombuffer(sub_data[sub_idx: sub_idx + 4], dtype=np.uint32)[
                    0]
                # print(frame_h)
                # print(sub_len)

            if type == 21:
                # frame_size = np.frombuffer(sub_data[sub_idx: sub_idx+4], dtype=np.uint16)[0]
                frame_size = \
                np.frombuffer(sub_data[sub_idx: sub_idx + 4], dtype=np.uint32)[
                    0]

            if type == 22:  # Формат картинки.
                img_type = sub_data[sub_idx: sub_idx + sub_len]
                # print(sub_len)
                # print('image_type:', repr(bytes(img_type).decode('utf-8')))

            if type == 23 and frame_w > 0 and frame_h > 0:  # Сама картинка.
                img_data = sub_data[sub_idx: sub_idx + sub_len]

                file_n += 1
                # print('file_n:', file_n)
                # print(sub_len)

                # Записываем изображение в файл
                # img_data.tofile("test/" + f'{file_n:06d}' + ".jpg")
            if type == 45:
                matrix_type = np.frombuffer(sub_data[sub_idx: sub_idx + sub_len],
                    dtype=np.double)[0]
                # print(matrix_type)
                licnums_by_frame['matrix_type'] = matrix_type

            if type == 46:
                focal_length = np.frombuffer(
                    sub_data[sub_idx:sub_idx + sub_len], dtype=np.double)[0]
                # print(focal_length)
                licnums_by_frame['focal_length'] = focal_length

            if type == 51:
                # print('matrix')
                info = np.frombuffer(sub_data[sub_idx: sub_idx + sub_len],
                                     dtype=np.byte)
                num_area = len(info) / (8 * 12)
                area_list = np.array_split(info, num_area)
                matrix3x4 = np.zeros((3, 4))
                ind = 0
                for area in area_list:
                    a_ij = np.frombuffer(area, dtype=np.int32)[0]
                    # print(a_ij)
                    matrix3x4[int(ind / 3)][ind % 4] = a_ij
                    ind += 1
                frame_info['matrix3x4'] = matrix3x4.tolist()

            sub_idx += sub_len
        # print('frame', frame_info)
        if 'radar_targets' in frame_info and 'licnums' in frame_info:
            licnums_by_frame['frames'].append(frame_info)
        idx += frame_len - 8
    # print(summ)
    return licnums_by_frame

if __name__ == '__main__':
    # show an "Open" dialog box and return the path to the selected file
    filename = askopenfilename()
    matrix_file = askopenfilename()


    # filename = "/Users/tina/Documents/Masters/diploma/video_speed_detection/data/2020.06.25_Office/1593092046_2020.06.25_16-34-06.rc"

    print('Pricessing file: ', filename)
    licnums = read_rc(filename)

    out_file_name = 'parsed/' + filename.split('/')[-1][:-3] + '.json'

    with open(matrix_file, 'r', encoding='utf-8') as f:
        matrix = json.load(f)
        print(matrix)
        licnums['matrix3x4'] = matrix

    with open(out_file_name, 'w', encoding='utf-8') as f:
        json.dump(licnums, f, indent=2)
