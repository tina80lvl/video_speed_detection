import numpy as np
import struct
import os
import io
from PIL import Image, ImageDraw
from datetime import datetime
from enum import IntEnum
import common
import codecs


class RC_FRAME(IntEnum):
    # Время съёмки кадра в UNIX формате secs (32 bit), usec(32 bit)
    CAPTURING_TIME = 0
    # ???
    CAPTURING_CLOCK = 1
    # ???
    CHECKSUM_MEASURING_BLOCK_BAD = 2
    # ???
    IDX = 3
    # Время в 90kHz тиках из FPGA
    TICKS = 4
    # Результаты распознавания номеров
    RECOGNIZE_REZULTS = 5
    # ???
    RECOGNIZE_RATING = 6
    # ???
    CLASSIFICATION_RESULT = 7
    # ???
    RADAR_TARGETS = 8
    # ???
    LAMP_FEATURES = 9
    # ???
    RAILWAY_STATE = 10
    # ???
    TL_CROSSWAY = 11
    # ???
    TIME_SYNCED = 12
    # ???
    TL_RECT = 13
    # Ширина оригинального кадра
    ORIG_WIDTH = 14
    # Высота оригинального кадра
    ORIG_HEIGHT = 15
    # Размер в байтах исходного изображения
    ORIG_LEN = 16
    # Формат оригинального кадра
    ORIG_FORMAT = 17
    # Сырые данные с радара (с АЦП)
    RADAR_RAW_DATA = 18
    # Ширина кадра, записанного в RC
    WIDTH = 19
    # Высота кадра, записанного в RC
    HEIGHT = 20
    # Размер в байтах сериализованного изображения
    LEN = 21
    # Формат кадра, записанного в RC
    # JPEG
    # GRAYSCALE
    # JPEG:NN (deprecated)
    FORMAT_DESC = 22
    # Данные картинки
    IMAGE = 23
    # Набор пар параметров (строка-строка)
    PARAMS_VALS = 24
    # ???
    OPTIMAL_GAMMA = 25
    # ???
    MATRIX = 26
    # ???
    BORDER_LINE_UP = 27
    # ???
    BORDER_LINE_DOWN = 28
    # ???
    CORDON_HEIGHT = 29
    # ???
    LASER_RAWDATA = 30
    # ???
    LASER_MEAS = 31
    # для стерео ???
    UNKNOWN_IGOR_32 = 32
    # для стерео ???
    UNKNOWN_IGOR_33 = 33
    # для стерео ???
    UNKNOWN_IGOR_34 = 34
    # ???
    RADAR_TARGETS_UNCORRECTED = 42
    # ???
    BLOCK_SERIAL = 43
    # Высота пикселя в микронах ???
    PIXEL_HEIGHT = 44
    # Ширина пикселя в микронах ???
    PIXEL_WIDTH = 45
    # ???
    REAL_ZOOM_POS = 46
    # Сырые данные от датчиков рельефа
    RAW_RELIEF_DATA = 47
    # Данные от отдельностоящего датчика рельефа
    RAW_RELIEF_DATA_SINGLE = 48
    # Данные от акселерометра-гироскопа
    RAW_IMU_DATA = 49
    # Скорость из OBD
    OBD_SPEED = 50


# ####### Helper functions ########
EXTENSIONS = [".rc"]
FOR_EXPORT = {
    RC_FRAME.RAW_IMU_DATA: ".accgyrob",
    RC_FRAME.OBD_SPEED: ".obdb",
    RC_FRAME.RAW_RELIEF_DATA_SINGLE: ".reliefb",
    RC_FRAME.RAW_RELIEF_DATA: ".reliefbb",
    RC_FRAME.RADAR_TARGETS: ".radarb"
}
RAW_IMU_DATA_HACK = True  # Ошибка в FPGA ядре - неправильный порядок данных в структуре


def get_bytes(buf, offset, dlen):
    tmp = None
    if type(buf) == bytes:
        if offset + dlen <= len(buf):
            tmp = buf[offset:offset + dlen]
    else:
        buf.seek(offset)
        tmp = buf.read(dlen)
        if len(tmp) != dlen:
            tmp = None
    return tmp


def floatfrombuf(buf, offset):
    f = None
    tmp = get_bytes(buf, offset, 4)
    if tmp is not None:
        f = struct.unpack('f', tmp)[0]
    return f


def doublefrombuf(buf, offset):
    f = None
    tmp = get_bytes(buf, offset, 8)
    if tmp is not None:
        f = struct.unpack('d', tmp)[0]
    return f


def int32frombuf(buf, offset):
    i = None
    tmp = get_bytes(buf, offset, 4)
    if tmp is not None:
        i = int.from_bytes(tmp, byteorder="little", signed=True)
    return i


def uint32frombuf(buf, offset):
    i = None
    tmp = get_bytes(buf, offset, 4)
    if tmp is not None:
        i = int.from_bytes(tmp, byteorder="little", signed=False)
    return i


def int16frombuf(buf, offset):
    i = None
    tmp = get_bytes(buf, offset, 2)
    if tmp is not None:
        i = int.from_bytes(tmp, byteorder="little", signed=True)
    return i


def uint16frombuf(buf, offset):
    i = None
    tmp = get_bytes(buf, offset, 2)
    if tmp is not None:
        i = int.from_bytes(tmp, byteorder="little", signed=False)
    return i


def txtformat(txt, frmt):
    try:
        otxt = frmt.format(txt)
    except:
        print('txtFormat error')
        otxt = 'error'
    return otxt


def str2float(numstr):
    try:
        ret = float(numstr)
    except:
        print('str2float error')
        ret = 0
    return ret


def findmagic(buf, offset, MAGIC=0x11223307):
    while True:
        magic = uint32frombuf(buf, offset)
        if magic is None:
            break
        elif magic == MAGIC:
            return offset
        offset += 1
    return None


def readavi(path):
    """
    Чтение avi файла
    """
    rc = []
    print(path)

    cap = cv2.VideoCapture(path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("video frames:", frame_count)

    fps = cap.get(cv2.CAP_PROP_FPS)
    interframe_ms = 1000 / fps
    print("video fps:", fps)

    print("processing...")
    for idx in range(frame_count):
        obj = {}
        fields = []
        # cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        a = 0
        lc = 0

        # ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        # print(idx, ts_ms)

        ts_ms = idx * interframe_ms
        secs = int(ts_ms / 1000)
        usecs = int(ts_ms * 1000) - 1000000 * int(ts_ms / 1000)
        fields.append({"id": RC_FRAME.CAPTURING_TIME, "offset": 0, "len": 0, "value": (secs, usecs)})
        # TEMPORARY!
        fields.append({"id": RC_FRAME.TICKS, "offset": 0, "len": 0, "value": (a, lc)})

        obj["fields"] = fields
        rc.append(obj)
    buf = None
    cap.release()
    return buf, rc


def get_sec_nsec_buf(buf):
    secs = int.from_bytes(buf[0:4], byteorder='little', signed=False)
    nsecs = int.from_bytes(buf[4:8], byteorder='little', signed=False)
    t = secs + nsecs / 1000000000
    return t


def get_sec_usec_buf(buf):
    secs = int.from_bytes(buf[0:4], byteorder='little', signed=False)
    nsecs = int.from_bytes(buf[4:8], byteorder='little', signed=False)
    t = secs + nsecs / 1000000
    return t


def parse_header(hdr):
    t = get_sec_usec_buf(hdr[0:8])
    ticks = int.from_bytes(hdr[8:12], byteorder='little', signed=False)
    dlen = int.from_bytes(hdr[12:16], byteorder='little', signed=False)
    return t, ticks, dlen


def split_data(fidx, to_extract):
    """
    Разбивка "rc" файла с данными разных типов на отдельные файлы
    :param fidx:
    fidx = {
        "type": f_type,
        "path": path,
        "index": idx
    }
    :param fields: field:ext
    :return:
    """
    fields = []
    for txtfield in to_extract:
        try:
            field = RC_FRAME.__members__[txtfield.split(".")[-1]]
            fields.append(field)
        except ValueError:
            continue

    new_paths = []
    path = fidx["path"]
    new_fields = []

    for field_id in fields:
        field_path = path + FOR_EXPORT[field_id]
        exists = os.path.exists(field_path)
        if not exists or (exists and (os.stat(field_path).st_size == 0)):
            new_fields.append(field_id)
        elif exists and (os.stat(field_path).st_size != 0):
            new_paths.append(path + FOR_EXPORT[field_id])

    if len(new_fields) == 0:
        print("all files already extracted")
        return new_paths

    with open(path, mode="rb") as f:
        new_files = {}
        for frame_fields in fidx["index"]["fields"]:
            for field_id in new_fields:
                if field_id in frame_fields:

                    '''sl = line.split("\t", 4)
                    if len(sl) < 4:
                        continue
                    secs = int(sl[0]) - 2208988800
                    if secs < 0:
                        print("bad timestamp", secs)
                        continue  # drop lines with bad timestamp
                    ticks = int(sl[1])
                    ltype = int(sl[2])
                    l = sl[3]'''
                    f.seek(frame_fields[RC_FRAME.CAPTURING_TIME]["offset"])
                    ct = f.read(8)
                    f.seek(frame_fields[RC_FRAME.TICKS]["offset"])
                    ticks = f.read(4)
                    f.seek(frame_fields[field_id]["offset"])
                    data = f.read(frame_fields[field_id]["len"])
                    if field_id not in new_files.keys():
                        new_files[field_id] = open(path + FOR_EXPORT[field_id], "wb+")
                        new_paths.append(path + FOR_EXPORT[field_id])

                    new_files[field_id].write(ct)
                    new_files[field_id].write(ticks)
                    new_files[field_id].write(int(frame_fields[field_id]["len"]).to_bytes(4, 'little', signed=False))
                    new_files[field_id].write(data)

    for f in new_files.values():
        f.close()

    return new_paths


def index_rc(path, add_fields={}, w_fields=True):
    cols = {"idx": [], "offset": [], "len": [], "fields": []}
    try:
        if len(add_fields) == 0:
            chunk_size = 1024
        else:
            chunk_size = 10 * 1024 * 1024
        with open(path, 'rb', buffering=chunk_size) as fid:
            print("indexing", path)
            cols = index_rc_io(fid, add_fields, w_fields)
    except Exception as e:
        print(e)
        print("error read file {}".format(path))
    return cols


def index_rc_io(fid, add_fields={}, w_fields=True):
    """
    Чтение rc файла, индексация фреймов. Можно добавить ещё столбцы с данными:
    time, ticks, ...
    [
        ?{
            padding //optional, 0-N bytes, normal zero bytes
        ?}
        {
            magic number, //current 0x11223307 by default
            total len, //all data plus total len field plus magic field
            [
                {
                    id, //two bytes field id (unique?)
                    len, //four bytes len of the payload
                    payload //data, 0?-N? bytes
                }
            ]
        }
    ]
    ?{
        padding //optional, 0-N bytes, normal zero bytes
    ?}
    """

    cols = {"idx": [], "offset": [], "len": [], "fields": []}

    addfields = {}
    if len(add_fields) != 0:
        for fld in add_fields:
            cols[fld] = []
            addfields[fld] = None

    idx = 0
    try:
        fid.seek(0, 2)
        f_size = fid.tell()
        fid.seek(0)
        print("f_size", f_size)
        while True:

            # Начинаем с поиска magic number
            # По-умолчанию magic number = 0x11223307
            curr_offset = fid.tell()
            magic_offset = findmagic(fid, curr_offset)

            # Если magic больше не найден
            if magic_offset is None:
                print(idx, "No more magic found!")
                return cols

            # Был ли разрыв (данные до magic buffer)?
            if magic_offset - curr_offset != 0:
                print(idx, "padding bytes before magic:".format(magic_offset - curr_offset))

            # Текущее смещение - после magic number
            curr_offset = fid.tell()

            # Общая длина
            total_l = uint32frombuf(fid, curr_offset)
            if total_l is None:
                print(idx,
                      "error! magic found, but no space for total len param. Got {} bytes, but needed 4".format(
                          f_size - curr_offset))
                return cols

            # Проверка общей длины
            if magic_offset + total_l > f_size:
                print(idx, "error! buffer overflow probably due to incorrect total len or unwanted end of file ?")
                return cols
            fields_len = total_l - 8
            curr_offset = fid.tell()
            # Опционально парсим данные и индексируем поля
            if len(add_fields) > 0 or w_fields:
                for fld in add_fields:
                    addfields[fld] = None
                fields = {}
                while fields_len > 0:
                    field_id = uint16frombuf(fid, curr_offset)
                    if field_id is None:
                        print("error! no space for field id. Got {} bytes, but needed 2".format(
                            f_size - curr_offset))
                        return cols

                    # print("id", field_id)
                    curr_offset = fid.tell()
                    fields_len -= 2
                    if fields_len < 0:
                        print("error! no space for field id according to total_l. Possible total_l is wrong")
                        return cols

                    field_len = uint32frombuf(fid, curr_offset)
                    if field_len is None:
                        print("error! no space for field_len. Got {} bytes, but needed 4".format(
                            f_size - curr_offset))
                        return cols

                    # print("len", field_len)
                    curr_offset = fid.tell()
                    fields_len -= 4
                    if fields_len < 0:
                        print("error! no space for field len according to total_l. Possible total_l is wrong")
                        return cols

                    if curr_offset + field_len > f_size:
                        print("error! buffer overflow probably due to incorrect field len or unwanted end of file?")
                        return cols

                    # bug workaround!
                    if field_id == RC_FRAME.LEN:
                        pic_len = uint32frombuf(fid, curr_offset)
                        if pic_len is None:
                            print("error! no space for pic_len. Got {} bytes, but needed 4".format(
                                f_size - curr_offset))
                            return cols

                    # bug workaround!
                    if field_id == RC_FRAME.IMAGE:
                        field_len = min(field_len, pic_len)

                    if curr_offset + field_len > f_size:
                        print("error! buffer overflow probably due to incorrect field len on unwanted end of file?")
                        return cols

                    # Вроде бы хорошее поле
                    fields[field_id] = {"offset": curr_offset, "len": field_len}

                    if field_id in add_fields:
                        if field_id == RC_FRAME.CAPTURING_TIME:
                            bbuf = get_bytes(fid, curr_offset, 8)
                            capt_time = get_frame_capturing_time(bbuf)
                            addfields[field_id] = capt_time[0] + capt_time[1] / 1000000

                    curr_offset += field_len
                    fields_len -= field_len
                    if fields_len < 0:
                        print(
                            "error! no space for field body according to total_l. Possible total_l or field_len is wrong")
                        return cols

                    fid.seek(curr_offset)

            # Добавляем в index
            cols["idx"].append(idx)
            cols["offset"].append(magic_offset)
            cols["len"].append(total_l)
            if len(add_fields) > 0:
                for fld in add_fields:
                    cols[fld].append(addfields[fld])
            cols["fields"].append(fields)
            idx += 1

            # Seek to next obj
            fid.seek(magic_offset + total_l)
    except Exception as e:
        print(e)
        print("error read file")
        return cols


def load_rc_idx(path, add_fields=[RC_FRAME.CAPTURING_TIME], force_index=False, tmp_path="./tmp/", only_with_idx=False):
    """
    Загрузка или перестроение индекса, при его отсутствии или нулевой длине

    :param path: путь к rc файлу

    :param add_fields: Дополнительные столбцы в индекс с распарсенными данными

    :return: возвращает словарь со столбцами idx, offset, len, [fields]
    """
    idx_path = common.get_hash_path(path, tmp_path)
    exists = os.path.exists(idx_path)
    import pickle
    if not exists or (exists and os.stat(idx_path).st_size == 0) or force_index:
        if only_with_idx:
            idx = None
            return idx
        idx = index_rc(path, add_fields)
        os.makedirs(os.path.dirname(idx_path), exist_ok=True)
        with open(idx_path, "wb") as f:
            pickle.dump(idx, f)
    else:
        with open(idx_path, "rb") as f:
            idx = pickle.load(f)
    idx[RC_FRAME.CAPTURING_TIME] = np.array(idx[RC_FRAME.CAPTURING_TIME])
    return idx


def parse_rc_chunk(rc_chunk_nfo, fid=None, buf=None, ffilter=[]):
    path = rc_chunk_nfo["path"]
    offset = rc_chunk_nfo["offset"]
    flen = rc_chunk_nfo["len"]
    fields = rc_chunk_nfo["fields"]
    ret = {}
    BUF_SIZE = 10 * 1024 * 1024
    try:
        if buf is None:
            if fid is None:
                fid = open(path, "rb", buffering=BUF_SIZE)
            fid.seek(offset)
            buf = fid.read(flen)

            for fld in fields.keys():
                if ffilter == [] or fld in ffilter:
                    f = fields[fld]
                    p_fld = get_field_data(buf[f["offset"] - offset: f["offset"] - offset + f["len"]], fld)
                    ret[fld] = p_fld

    except Exception as e:
        print("error working with file", path, e)
        return None
    return ret


def readrc(path):
    """
    Чтение rc файла в память, разбор по полям
    [
        ?{
            padding //optional, 0-N bytes, normal zero bytes
        ?}
        {
            magic number, //current 0x11223307 by default
            total len, //all data plus total len field plus magic field
            [
                {
                    id, //two bytes field id (unique?)
                    len, //four bytes len of the payload
                    payload //data, 0?-N? bytes
                }
            ]
        }
    ]
    ?{
        padding //optional, 0-N bytes, normal zero bytes
    ?}
    """

    rc = []
    curr_offset = 0

    try:
        with open(path, 'rb') as fid:
            buf = fid.read()
        print("readed")
    except:
        print("error open file {}".format(path))
        return None, None

    idx = 0
    while True:
        # print(idx)
        obj = {}
        obj["fields"] = []

        # Начинаем с поиска magic number
        # По-умолчанию magic number = 0x11223307
        magic_offset = findmagic(buf, curr_offset)

        # Если magic больше не найден
        if magic_offset is None:
            obj["offset"] = curr_offset
            obj["total len"] = len(buf) - curr_offset
            if obj["total len"] > 0:
                # rc.append(obj)
                pass
            print("No more magic found!")
            return buf, rc

        # Был ли разрыв (данные до magic buffer)?
        if magic_offset - curr_offset != 0:
            obj["offset"] = curr_offset
            obj["total len"] = magic_offset - curr_offset
            # rc.append(obj)
            print("padding bytes before magic:".format(magic_offset - curr_offset))

        # Смещение magic number
        obj["offset"] = magic_offset
        # Текущее смещение - после magic number
        curr_offset = magic_offset + 4

        # Общая длина
        total_l = uint32frombuf(buf, curr_offset)
        if total_l is None:
            print("error! magic found, but no space for total len param. Got {} bytes, but needed 4".format(
                len(buf) - curr_offset))
            obj["total len"] = len(buf) - magic_offset
            # rc.append(obj)
            return buf, rc

        # Проверка общей длины
        if magic_offset + total_l > len(buf):
            print("error! buffer overflow probably due to incorrect total len or unwanted end of file ?")
            obj["total len"] = len(buf) - magic_offset
            # rc.append(obj)
            return buf, rc

        obj["total len"] = total_l
        # print("total_l=", total_l)
        # Текущее смещение - после общей длины
        curr_offset += 4
        # Полезная нагрузка = общая длина - magic number - длина
        total_l -= 8

        # Десериализация полей
        fields = []  # FIXME field id is unique?
        while total_l > 0:
            field_id = uint16frombuf(buf, curr_offset)
            if field_id is None:
                print("error! no space for field id. Got {} bytes, but needed 2".format(len(buf) - curr_offset))
                obj["total len"] = len(buf) - magic_offset
                # rc.append(obj)
                return buf, rc

            # print("id", field_id)
            curr_offset += 2
            total_l -= 2
            if total_l < 0:
                print("error! no space for field id according to total_l. Possible total_l is wrong")
                obj["total len"] = len(buf) - magic_offset
                # rc.append(obj)
                return buf, rc

            field_len = uint32frombuf(buf, curr_offset)
            if field_len is None:
                print("error! no space for field_len. Got {} bytes, but needed 4".format(len(buf) - curr_offset))
                obj["total len"] = len(buf) - magic_offset
                # rc.append(obj)
                return buf, rc

            # print("len", field_len)
            curr_offset += 4
            total_l -= 4
            if total_l < 0:
                print("error! no space for field len according to total_l. Possible total_l is wrong")
                obj["total len"] = len(buf) - magic_offset
                # rc.append(obj)
                return buf, rc

            if curr_offset + field_len > len(buf):
                print("error! buffer overflow probably due to incorrect field len or unwanted end of file?")
                obj["total len"] = len(buf) - magic_offset
                # rc.append(obj)
                return buf, rc

            # bug workaround!
            if field_id == RC_FRAME.LEN:
                pic_len = uint32frombuf(buf, curr_offset)
                if pic_len is None:
                    print("error! no space for pic_len. Got {} bytes, but needed 4".format(len(buf) - curr_offset))
                    obj["total len"] = len(buf) - magic_offset
                    # rc.append(obj)
                    return buf, rc

            # bug workaround!
            if field_id == RC_FRAME.IMAGE:
                field_len = min(field_len, pic_len)

            if curr_offset + field_len > len(buf):
                print("error! buffer overflow probably due to incorrect field len on unwanted end of file?")
                obj["total len"] = len(buf) - magic_offset
                # rc.append(obj)
                return buf, rc

            # Вроде бы хорошее поле
            fields.append({"id": field_id, "offset": curr_offset, "len": field_len})

            curr_offset += field_len
            total_l -= field_len
            if total_l < 0:
                print("error! no space for field body according to total_l. Possible total_l or field_len is wrong")
                obj["total len"] = len(buf) - magic_offset
                # rc.append(obj)
                return buf, rc

        # print("good")
        obj["fields"] = fields
        rc.append(obj)

        idx += 1


def synchronizelr(bufl, rcl, bufr, rcr):
    """
    Поиск в двух rc буферах синхронных кадров
    """
    deltaframes = 3
    deltaticks = 4
    for idxl, objl in enumerate(rcl):
        # print("-")
        tickl = getfielddata(bufl, objl["fields"], RC_FRAME.TICKS)
        start = idxl - deltaframes
        stop = idxl + deltaframes
        if start < 0:
            start = 0
        if stop >= len(rcr):
            stop = len(rcr) - 1
        objsr = rcr[start:stop]
        objl["pair"] = None
        for objr in objsr:
            # print(".")
            tickr = getfielddata(bufr, objr["fields"], RC_FRAME.TICKS)
            # print(tickl, tickr, abs(tickl-tickr))
            if abs(tickl - tickr) < deltaticks:
                objl["pair"] = objr
                break

    for idxr, objr in enumerate(rcr):
        tickr = getfielddata(bufr, objr["fields"], RC_FRAME.TICKS)
        start = idxr - deltaframes
        stop = idxr + deltaframes
        if start < 0:
            start = 0
        if stop >= len(rcl):
            stop = len(rcl) - 1
        objsl = rcl[start:stop]
        objr["pair"] = None
        for objl in objsl:
            tickl = getfielddata(bufl, objl["fields"], RC_FRAME.TICKS)
            if abs(tickl - tickr) < deltaticks:
                objr["pair"] = objl
                break


def get_frame_capturing_time(buf):
    """
    RC_FRAME_CAPTURING_TIME = 0
    Время съёмки кадра в UNIX формате secs (32 bit), usec(32 bit)
    """
    ret = (uint32frombuf(buf, 0), uint32frombuf(buf, 4))
    # (sec, usec)
    return ret


def get_frame_capturing_clock(buf):
    """
    RC_FRAME_CAPTURING_CLOCK = 1
    Время съёмки кадра в формате secs (32 bit), nsec(32 bit)
    """
    ret = (uint32frombuf(buf, 0), uint32frombuf(buf, 4))
    # (sec, nsec)
    return ret


def get_frame_idx(buf):
    """
    RC_FRAME_IDX = 3
    Счётчик конца фреймов
    """
    ret = uint32frombuf(buf, 0)
    return ret


def get_frame_ticks(buf):
    """
    RC_FRAME_TICKS = 4
    Время в 90kHz тиках из FPGA
    """
    ret = uint32frombuf(buf, 0)
    return ret


def get_recognition_results(buf):
    """
    RC_FRAME_RECOGNIZE_REZULTS = 5
    Результаты распознавания номеров
    """
    ret = []
    i = 0
    while i < len(buf):
        el = {}
        # Код страны + номер внутри страны FIXME
        el["numformat"] = int32frombuf(buf, i)
        i += 4
        # Число символов в номере
        el["nsymbols"] = int32frombuf(buf, i)
        i += 4
        # Распознанный номер
        # print(buf[i:i + 16 * 2])
        try:
            el["text16"] = buf[i:i + 16 * 2].decode("utf-16")
        except:
            el["text16"] = 'exception'
        i += 16 * 2
        # Общая оценка качества распоснавания
        el["allcert"] = int16frombuf(buf, i)
        i += 2
        # Оценка качества распоснавания каждой буковки
        # el["certlist"] = buf[i:i + 16 * 2].decode("utf-16")
        i += 16 * 2
        # Координаты прямоугольника с номером
        x = []
        for k in range(4):
            x.append(int16frombuf(buf, i))
            i += 2
        el["x"] = x
        y = []
        for k in range(4):
            y.append(int16frombuf(buf, i))
            i += 2
        el["y"] = y
        i += 2  # FIXME

        ret.append(el)
    return ret


def get_radar_targets(buf):
    """
    RADAR_TARGETS = 8

    uint32_t num_elem = curLen / sizeof(umrr_target);
    for (uint32_t i = 0; i < num_elem; i++) {
        umrr_target *elem = new umrr_target;
        ptr = dataFromBuf(ptr, *elem);
        v[elem->id] = elem;
    }

    int id;
    double x;
    double y;
    double xspeed;
    double yspeed;
    double len;

    // unused values, only for sizeof(umrr_target) not changed
    double imgx_unused;
    double imgy_unused;
    int numw_unused;

    double imgx_left_unused;
    double imgx_right_unused;

    double imgy_top_unused;
    double imgy_bottom_unused;

    :return:
    """

    UMMR_TARGET_LENGH = 11 * 8 + 2 * 4
    ret = {}
    i = 0
    while i < len(buf):
        id = uint32frombuf(buf, i)
        i += 4
        x = doublefrombuf(buf, i)
        i += 8
        y = doublefrombuf(buf, i)
        i += 8
        xspeed = doublefrombuf(buf, i)
        i += 8
        yspeed = doublefrombuf(buf, i)
        i += 8
        dlen = doublefrombuf(buf, i)
        i += 8

        i += 8
        i += 8
        i += 4
        i += 8
        i += 8
        i += 8
        i += 8

        ret[id] = {"x": x, "y": y, "xspeed": xspeed, "yspeed": yspeed, "len": dlen}
    return ret


def get_orig_width(buf):
    """
    ORIG_WIDTH = 14
    :return:
    """
    return int32frombuf(buf, 0)


def get_orig_height(buf):
    """
    ORIG_HEIGHT = 15
    :return:
    """
    return int32frombuf(buf, 0)


def get_frame_width(buf):
    """
    RC_FRAME_WIDTH = 19
    Ширина кадра, записанного в RC
    """
    return int32frombuf(buf, 0)


def get_frame_height(buf):
    """
    RC_FRAME_HEIGHT = 20
    Высота кадра, записанного в RC
    """
    return int32frombuf(buf, 0)


def get_frame_format_desc(buf):
    """
    RC_FRAME_FORMAT_DESC = 22
    - JPEG
    - GRAYSCALE
    - JPEG:75 (JPEG:NN) Deprecated
    ???
    """
    return buf.decode('utf-8')


def get_frame_image(buf):
    """
    RC_FRAME_IMAGE = 23
    - JPEG file
    - grayscale image W x H x 8bit
    ???
    """
    return buf


def get_params_vals(buf):
    """
    RC_FRAME_PARAMS_VALS = 24
    Набор параметров (ключ (строка)-значение (строка))
    """
    ret = {}
    i = 0
    while i < len(buf):
        plen = uint32frombuf(buf, i)
        i += 4
        par = buf[i:i + plen].decode('utf-8')
        i += plen
        vlen = uint32frombuf(buf, i)
        i += 4
        val = buf[i:i + vlen].decode('utf-8')
        i += vlen
        ret[par] = val
    return ret


def get_raw_laser_data(buf):
    """
    RC_FRAME_LASER_RAWDATA = 30
    """
    DLEN = 8192
    OFFSET = 128
    arr = np.int16(np.frombuffer(buf, dtype=np.uint8).copy()) - OFFSET
    for i in range(int(len(buf) / DLEN)):
        # FIXME what in this 8 bytes?
        arr[range(i * DLEN, i * DLEN + 8)] = 0
    return arr


def get_laser_meas(buf):
    """
    RC_FRAME_LASER_MEAS = 31
    """
    ret = dict()
    ret["range_m"] = floatfrombuf(buf, 0)
    ret["err_range_m"] = floatfrombuf(buf, 4)
    ret["speed_kmh"] = int32frombuf(buf, 8)
    ret["err_speed_kmh"] = floatfrombuf(buf, 12)
    return ret


def get_pixel_height(buf):
    """
    PIXEL_HEIGHT = 44
    :return:
    """
    ret = doublefrombuf(buf, 0)
    return ret


def get_pixel_width(buf):
    """
    PIXEL_WIDTH = 45
    :return:
    """
    ret = doublefrombuf(buf, 0)
    return ret


def get_real_zoom_pos(buf):
    """
    REAL_ZOOM_POS = 46
    :param buf:
    :return:
    """
    ret = doublefrombuf(buf, 0)
    return ret


def get_raw_relief_data(buf):
    """
    RAW_RELIEF_DATA = 47
    """
    ret = []
    i = 0
    while i < len(buf):
        plen = uint32frombuf(buf, i)
        i += 4
        par = buf[i:i + plen]
        i += plen
        ret.append(par)
    return ret


def get_raw_relief_data_single(buf):
    """
    RAW_RELIEF_DATA_SINGLE = 48

    struct relief_serial_data_48_s
    {
      uint32_t ticks; // wr on parkonavt
      uint32_t timestamp; // count from relief
      uint16_t a;
      uint16_t b;
    } attribute ((packed));
    """
    ret = dict()
    ret["ticks"] = np.frombuffer(buf, dtype=np.uint32)[::3]
    ret["counter"] = np.frombuffer(buf, dtype=np.uint32)[1::3]
    ret["raising"] = np.frombuffer(buf, dtype=np.uint16)[4::6]
    ret["falling"] = np.frombuffer(buf, dtype=np.uint16)[5::6]

    return ret


def get_raw_imu_data(buf):
    """
    RAW_IMU_DATA = 49
    //LSM6DSM
    struct imuspi_data_49_s
    {
        uint32_t ticks; // wr on parkonavt
        int16_t gyro_x;
        int16_t gyro_y;
        int16_t gyro_z;
        int16_t accel_x;
        int16_t accel_y;
        int16_t accel_z;
    } attribute ((packed));
    """
    ret = dict()
    if RAW_IMU_DATA_HACK:
        ret["ticks"] = np.frombuffer(buf, dtype=np.uint32)[::4]
        ret["gyro_y"] = np.frombuffer(buf, dtype=np.int16)[2::8]
        ret["gyro_x"] = np.frombuffer(buf, dtype=np.int16)[3::8]
        ret["accel_x"] = np.frombuffer(buf, dtype=np.int16)[4::8]
        ret["gyro_z"] = np.frombuffer(buf, dtype=np.int16)[5::8]
        ret["accel_z"] = np.frombuffer(buf, dtype=np.int16)[6::8]
        ret["accel_y"] = np.frombuffer(buf, dtype=np.int16)[7::8]
    else:
        ret["ticks"] = np.frombuffer(buf, dtype=np.uint32)[::4]
        ret["gyro_x"] = np.frombuffer(buf, dtype=np.int16)[2::8]
        ret["gyro_y"] = np.frombuffer(buf, dtype=np.int16)[3::8]
        ret["gyro_z"] = np.frombuffer(buf, dtype=np.int16)[4::8]
        ret["accel_x"] = np.frombuffer(buf, dtype=np.int16)[5::8]
        ret["accel_y"] = np.frombuffer(buf, dtype=np.int16)[6::8]
        ret["accel_z"] = np.frombuffer(buf, dtype=np.int16)[7::8]
    return ret


def get_obd_speed(buf):
    """
    OBD_SPEED = 50
    struct obd_data_50_s
    {
        uint32_t ticks; //
        float speed; // speed im km/h
    } attribute ((packed));
    """
    ret = dict()
    ret["ticks"] = np.frombuffer(buf, dtype=np.uint32)[::2]
    ret["speed"] = np.frombuffer(buf, dtype=np.float32)[1::2]
    return ret


def get_field_data(field_body, fieldid):
    """
    Десериализация поля по его ID
    """
    ret = None
    if fieldid == RC_FRAME.IMAGE:
        ret = get_frame_image(field_body)
    elif fieldid == RC_FRAME.CAPTURING_TIME:
        ret = get_frame_capturing_time(field_body)
    elif fieldid == RC_FRAME.CAPTURING_CLOCK:
        ret = get_frame_capturing_clock(field_body)
    elif fieldid == RC_FRAME.IDX:
        ret = get_frame_idx(field_body)
    elif fieldid == RC_FRAME.TICKS:
        ret = get_frame_ticks(field_body)
    elif fieldid == RC_FRAME.WIDTH:
        ret = get_frame_width(field_body)
    elif fieldid == RC_FRAME.ORIG_HEIGHT:
        ret = get_orig_height(field_body)
    elif fieldid == RC_FRAME.ORIG_WIDTH:
        ret = get_orig_width(field_body)
    elif fieldid == RC_FRAME.PIXEL_HEIGHT:
        ret = get_pixel_height(field_body)
    elif fieldid == RC_FRAME.PIXEL_WIDTH:
        ret = get_pixel_width(field_body)
    elif fieldid == RC_FRAME.REAL_ZOOM_POS:
        ret = get_real_zoom_pos(field_body)
    elif fieldid == RC_FRAME.RADAR_TARGETS:
        ret = get_radar_targets(field_body)
    elif fieldid == RC_FRAME.HEIGHT:
        ret = get_frame_height(field_body)
    elif fieldid == RC_FRAME.FORMAT_DESC:
        ret = get_frame_format_desc(field_body)
    elif fieldid == RC_FRAME.PARAMS_VALS:
        ret = get_params_vals(field_body)
    elif fieldid == RC_FRAME.LASER_MEAS:
        ret = get_laser_meas(field_body)
    elif fieldid == RC_FRAME.RECOGNIZE_REZULTS:
        ret = get_recognition_results(field_body)
    elif fieldid == RC_FRAME.LASER_RAWDATA:
        ret = get_raw_laser_data(field_body)
    elif fieldid == RC_FRAME.RAW_RELIEF_DATA:
        ret = get_raw_relief_data(field_body)
    elif fieldid == RC_FRAME.RAW_RELIEF_DATA_SINGLE:
        ret = get_raw_relief_data_single(field_body)
    elif fieldid == RC_FRAME.RAW_IMU_DATA:
        ret = get_raw_imu_data(field_body)
    elif fieldid == RC_FRAME.OBD_SPEED:
        ret = get_obd_speed(field_body)
    else:
        pass
        # print("This field ID does not supported yet!", fieldid)
    return ret


def getfielddata(buf, fields, fieldid):
    """
    Десериализация поля по его ID
    """
    ret = None
    for fld in fields:
        if fld['id'] == fieldid:
            if 'value' in fld:
                ret = fld['value']
            else:
                field_body = buf[fld["offset"]:fld["offset"] + fld["len"]]
            if fieldid == RC_FRAME.IMAGE:
                ret = get_frame_image(field_body)
            elif fieldid == RC_FRAME.CAPTURING_TIME:
                ret = get_frame_capturing_time(field_body)
            elif fieldid == RC_FRAME.CAPTURING_CLOCK:
                ret = get_frame_capturing_clock(field_body)
            elif fieldid == RC_FRAME.IDX:
                ret = get_frame_idx(field_body)
            elif fieldid == RC_FRAME.TICKS:
                ret = get_frame_ticks(field_body)
            elif fieldid == RC_FRAME.WIDTH:
                ret = get_frame_width(field_body)
            elif fieldid == RC_FRAME.RADAR_TARGETS:
                ret = get_radar_targets(field_body)
            elif fieldid == RC_FRAME.HEIGHT:
                ret = get_frame_height(field_body)
            elif fieldid == RC_FRAME.ORIG_HEIGHT:
                ret = get_orig_height(field_body)
            elif fieldid == RC_FRAME.ORIG_WIDTH:
                ret = get_orig_width(field_body)
            elif fieldid == RC_FRAME.PIXEL_HEIGHT:
                ret = get_pixel_height(field_body)
            elif fieldid == RC_FRAME.PIXEL_WIDTH:
                ret = get_pixel_width(field_body)
            elif fieldid == RC_FRAME.REAL_ZOOM_POS:
                ret = get_real_zoom_pos(field_body)
            elif fieldid == RC_FRAME.FORMAT_DESC:
                ret = get_frame_format_desc(field_body)
            elif fieldid == RC_FRAME.PARAMS_VALS:
                ret = get_params_vals(field_body)
            elif fieldid == RC_FRAME.LASER_MEAS:
                ret = get_laser_meas(field_body)
            elif fieldid == RC_FRAME.RECOGNIZE_REZULTS:
                ret = get_recognition_results(field_body)
            elif fieldid == RC_FRAME.LASER_RAWDATA:
                ret = get_raw_laser_data(field_body)
            elif fieldid == RC_FRAME.RAW_RELIEF_DATA:
                ret = get_raw_relief_data(field_body)
            elif fieldid == RC_FRAME.RAW_RELIEF_DATA_SINGLE:
                ret = get_raw_relief_data_single(field_body)
            elif fieldid == RC_FRAME.RAW_IMU_DATA:
                ret = get_raw_imu_data(field_body)
            elif fieldid == RC_FRAME.OBD_SPEED:
                ret = get_obd_speed(field_body)
            else:
                print("This field ID does not supported yet!", fieldid)

            # Обрабатываем только первое вхождение поля FIXME
            break
    return ret


def getfield(fields, fieldid):
    for field in fields:
        if field["id"] == fieldid:
            return field
    return None


def getimage(buf, rc, index, obj=None, needed=None):
    if obj is None:
        obj = rc[index]
    img = {}
    fields = obj["fields"]

    if needed is None:
        needed = [
            RC_FRAME.IMAGE,
            RC_FRAME.TICKS,
            RC_FRAME.PARAMS_VALS,
            RC_FRAME.LASER_MEAS,
            RC_FRAME.RECOGNIZE_REZULTS,
            RC_FRAME.LASER_RAWDATA,
            RC_FRAME.CAPTURING_TIME,
            RC_FRAME.FORMAT_DESC,
            RC_FRAME.WIDTH,
            RC_FRAME.HEIGH,
            RC_FRAME.RAW_RELIEF_DATA
        ]
    for need in needed:
        ret = getfielddata(buf, fields, need)
        if ret is not None:
            # if need == RC_FRAME.PARAMS_VALS:
            #    print("Got GPS data", index, ret)
            img[need] = ret

    # if "pair" in obj:
    img["pair"] = None  # obj["pair"]
    # else:
    #    img["pair"] = None
    return img


def saveimage(buf, rc, index, obj=None, fn=None):
    if obj is None:
        obj = rc[index]
    img = {}
    fields = obj["fields"]

    needed = [
        RC_FRAME.IMAGE
    ]
    for need in needed:
        ret = getfielddata(buf, fields, need)
        if ret is not None:
            img[need] = ret
    if fn is None:
        fn = str(index) + ".jpg"
    with open(fn, "wb") as f:
        f.write(img[RC_FRAME.IMAGE])


def save_video(fts, fn, track, t_beg, t_end, VIDEO_FPS=25, w=1920, h=1080, isColor=False):
    import cv2
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'JPEG')
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter(fn, fourcc, VIDEO_FPS, (w, h), isColor=isColor)

    tr_files = fts.get_track_indexes_by_range(track, t_beg, t_end)
    if len(tr_files) > 0:
        for i, f_desc in enumerate(tr_files):
            # f_desc = [tr_file_f, f_idx_f, idx_beg, idx_end]
            fn = f_desc["file_desc"]["path"]
            idx_f = f_desc["file_desc"]["index"]
            idx_beg = f_desc["beg_index"]
            idx_end = f_desc["end_index"]
            with open(fn, "rb") as fid:
                for fridx in range(idx_beg, idx_end):

                    chunk = {
                        "path": fn,
                        "offset": idx_f["offset"][fridx],
                        "len": idx_f["len"][fridx],
                        "fields": idx_f["fields"][fridx]
                    }
                    tim = idx_f[RC_FRAME.CAPTURING_TIME][fridx]

                    ffilter = []
                    d = parse_rc_chunk(chunk, fid=fid, buf=None, ffilter=ffilter)
                    w = d[RC_FRAME.WIDTH]
                    h = d[RC_FRAME.HEIGHT]
                    img = d.get(RC_FRAME.IMAGE)
                    fmt = d.get(RC_FRAME.FORMAT_DESC)
                    imagel = None
                    if fmt == "GRAYSCALE":
                        # Выводим несжатую картинку
                        imagel = np.frombuffer(img, dtype=np.uint8).reshape(h, w).T[:, ::-1]
                    elif fmt == "JPEG":
                        # Выводим картинку из JPEG
                        # pil_image = Image.open(io.BytesIO(img)).rotate(-90, expand=1)
                        pil_image = Image.open(io.BytesIO(img))

                        draw = ImageDraw.Draw(pil_image)
                        ts = datetime.utcfromtimestamp(tim + 3 * 60 * 60).strftime('%H:%M:%S:%f')[:-3]
                        draw.text((5, 5), ts, fill=255)  # , font=ImageFont.truetype("font_path123"))

                        imagel = np.asarray(pil_image, dtype="uint8")
                        # if len(imagel.shape) == 3:
                        #    pass
                        # else:
                        #    imagel = np.stack((imagel,) * 3, -1)
                    if imagel is not None:
                        print(fn, fridx)

                        if len(imagel.shape) == 3:
                            frame = imagel.reshape((h, w, 3))
                        else:
                            frame = imagel.reshape((h, w))
                        frame = imagel
                        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        out.write(frame)

                        cv2.imshow('frame', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

        out.release()
        cv2.destroyAllWindows()
