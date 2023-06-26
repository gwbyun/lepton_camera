#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 15:02:59 2023

@author: gw"""


# 버퍼 크기가 16인 문자열 버퍼 
from ctypes import *
from multiprocessing  import Queue
import platform
import ctypes
import gc
import numpy as np
import h5py
from tifffile import imsave
import cv2
import threading

# 플랫폼에 따라 라이브러리 파일 경로 설정
if platform.system() == "Windows":
    libuvc = CDLL("libuvc.dll")
elif platform.system() == "Linux":
    libuvc = CDLL("libuvc.so")
elif platform.system() == "Darwin":
    libuvc = CDLL("libuvc.dylib")
else:
    raise OSError("Unsupported platform")

# libuvc.h 헤더 파일의 필요한 정의들

# UVC 컨텍스트 구조체 정의
BUF_SIZE = 2
q = Queue(BUF_SIZE)
colorMapType = 0

class uvc_context(Structure):
  _fields_ = [("usb_ctx", c_void_p),
              ("own_usb_ctx", c_uint8),
              ("open_devices", c_void_p),
              ("handler_thread", c_ulong),
              ("kill_handler_thread", c_int)]
# UVC 장치 구조체 정의f
class uvc_device(Structure):
  _fields_ = [("ctx", POINTER(uvc_context)),
              ("ref", c_int),
              ("usb_dev", c_void_p)]

class uvc_device_handle(Structure):
  _fields_ = [("dev", POINTER(uvc_device)),
              ("prev", c_void_p),
              ("next", c_void_p),
              ("usb_devh", c_void_p),
              ("info", c_void_p),
              ("status_xfer", c_void_p),
              ("status_buf", c_ubyte * 32),
              ("status_cb", c_void_p),
              ("status_user_ptr", c_void_p),
              ("button_cb", c_void_p),
              ("button_user_ptr", c_void_p),
              ("streams", c_void_p),
              ("is_isight", c_ubyte)]

class uvc_stream_ctrl(Structure):
  _fields_ = [("bmHint", c_uint16),
              ("bFormatIndex", c_uint8),
              ("bFrameIndex", c_uint8),
              ("dwFrameInterval", c_uint32),
              ("wKeyFrameRate", c_uint16),
              ("wPFrameRate", c_uint16),
              ("wCompQuality", c_uint16),
              ("wCompWindowSize", c_uint16),
              ("wDelay", c_uint16),
              ("dwMaxVideoFrameSize", c_uint32),
              ("dwMaxPayloadTransferSize", c_uint32),
              ("dwClockFrequency", c_uint32),
              ("bmFramingInfo", c_uint8),
              ("bPreferredVersion", c_uint8),
              ("bMinVersion", c_uint8),
              ("bMaxVersion", c_uint8),
              ("bInterfaceNumber", c_uint8)]
def print_device_info(devh):
  vers = lep_oem_sw_version()
  call_extension_unit(devh, OEM_UNIT_ID, 9, byref(vers), 8)
  print("Version gpp: {0}.{1}.{2} dsp: {3}.{4}.{5}".format(
    vers.gpp_major, vers.gpp_minor, vers.gpp_build,
    vers.dsp_major, vers.dsp_minor, vers.dsp_build,
  ))
def uvc_iter_formats(devh):
  p_format_desc = libuvc.uvc_get_format_descs(devh)
  while p_format_desc:
    yield p_format_desc.contents
    p_format_desc = p_format_desc.contents.next

def uvc_iter_frames_for_format(devh, format_desc):
  p_frame_desc = format_desc.frame_descs
  while p_frame_desc:
    yield p_frame_desc.contents
    p_frame_desc = p_frame_desc.contents.next

def print_device_formats(devh):
  for format_desc in uvc_iter_formats(devh):
    print("format: {0}".format(format_desc.guidFormat[0:4]))
    for frame_desc in uvc_iter_frames_for_format(devh, format_desc):
      print("  frame {0}x{1} @ {2}fps".format(frame_desc.wWidth, frame_desc.wHeight, int(1e7 / frame_desc.dwDefaultFrameInterval)))

def uvc_get_frame_formats_by_guid(devh, vs_fmt_guid):
  for format_desc in uvc_iter_formats(devh):
    if vs_fmt_guid[0:4] == format_desc.guidFormat[0:4]:
      return [fmt for fmt in uvc_iter_frames_for_format(devh, format_desc)]
  return []
def call_extension_unit(devh, unit, control, data, size):
  return libuvc.uvc_get_ctrl(devh, unit, control, data, size, 0x81)

class uvc_format_desc(Structure):
  pass
class uvc_frame_desc(Structure):
  pass

uvc_frame_desc._fields_ = [
              ("parent", POINTER(uvc_format_desc)),
              ("prev", POINTER(uvc_frame_desc)),
              ("next", POINTER(uvc_frame_desc)),
              # /** Type of frame, such as JPEG frame or uncompressed frme */
              ("bDescriptorSubtype", c_uint), # enum uvc_vs_desc_subtype bDescriptorSubtype;
              # /** Index of the frame within the list of specs available for this format */
              ("bFrameIndex", c_uint8),
              ("bmCapabilities", c_uint8),
              # /** Image width */
              ("wWidth", c_uint16),
              # /** Image height */
              ("wHeight", c_uint16),
              # /** Bitrate of corresponding stream at minimal frame rate */
              ("dwMinBitRate", c_uint32),
              # /** Bitrate of corresponding stream at maximal frame rate */
              ("dwMaxBitRate", c_uint32),
              # /** Maximum number of bytes for a video frame */
              ("dwMaxVideoFrameBufferSize", c_uint32),
              # /** Default frame interval (in 100ns units) */
              ("dwDefaultFrameInterval", c_uint32),
              # /** Minimum frame interval for continuous mode (100ns units) */
              ("dwMinFrameInterval", c_uint32),
              # /** Maximum frame interval for continuous mode (100ns units) */
              ("dwMaxFrameInterval", c_uint32),
              # /** Granularity of frame interval range for continuous mode (100ns) */
              ("dwFrameIntervalStep", c_uint32),
              # /** Frame intervals */
              ("bFrameIntervalType", c_uint8),
              # /** number of bytes per line */
              ("dwBytesPerLine", c_uint32),
              # /** Available frame rates, zero-terminated (in 100ns units) */
              ("intervals", POINTER(c_uint32))]

uvc_format_desc._fields_ = [
              ("parent", c_void_p),
              ("prev", POINTER(uvc_format_desc)),
              ("next", POINTER(uvc_format_desc)),
              # /** Type of image stream, such as JPEG or uncompressed. */
              ("bDescriptorSubtype", c_uint), # enum uvc_vs_desc_subtype bDescriptorSubtype;
              # /** Identifier of this format within the VS interface's format list */
              ("bFormatIndex", c_uint8),
              ("bNumFrameDescriptors", c_uint8),
              # /** Format specifier */
              ("guidFormat", c_char * 16), # union { uint8_t guidFormat[16]; uint8_t fourccFormat[4]; }
              # /** Format-specific data */
              ("bBitsPerPixel", c_uint8),
              # /** Default {uvc_frame_desc} to choose given this format */
              ("bDefaultFrameIndex", c_uint8),
              ("bAspectRatioX", c_uint8),
              ("bAspectRatioY", c_uint8),
              ("bmInterlaceFlags", c_uint8),
              ("bCopyProtect", c_uint8),
              ("bVariableSize", c_uint8),
              # /** Available frame specifications for this format */
              ("frame_descs", POINTER(uvc_frame_desc))]



class lep_oem_sw_version(Structure):
  _fields_ = [("gpp_major", c_ubyte),
              ("gpp_minor", c_ubyte),
              ("gpp_build", c_ubyte),
              ("dsp_major", c_ubyte),
              ("dsp_minor", c_ubyte),
              ("dsp_build", c_ubyte),
              ("reserved", c_ushort)]
''' 
  # UVC 컨텍스트 포인터 정의
ctx= POINTER(uvc_context)
dev = POINTER(uvc_device)
devh = POINTER(uvc_device_handle)
'''



class timeval(Structure):
  _fields_ = [("tv_sec", c_long), ("tv_usec", c_long)]
class uvc_frame(Structure):
  _fields_ = [# /** Image data for this frame */
              ("data", POINTER(c_uint8)),
              # /** Size of image data buffer */
              ("data_bytes", c_size_t),
              # /** Width of image in pixels */
              ("width", c_uint32),
              # /** Height of image in pixels */
              ("height", c_uint32),
              # /** Pixel data format */
              ("frame_format", c_uint), # enum uvc_frame_format frame_format
              # /** Number of bytes per horizontal line (undefined for compressed format) */
              ("step", c_size_t),
              # /** Frame number (may skip, but is strictly monotonically increasing) */
              ("sequence", c_uint32),
              # /** Estimate of system time when the device started capturing the image */
              ("capture_time", timeval),
              # /** Handle on the device that produced the image.
              #  * @warning You must not call any uvc_* functions during a callback. */
              ("source", POINTER(uvc_device)),
              # /** Is the data buffer owned by the library?
              #  * If 1, the data buffer can be arbitrarily reallocated by frame conversion
              #  * functions.
              #  * If 0, the data buffer will not be reallocated or freed by the library.
              #  * Set this field to zero if you are supplying the buffer.
              #  */
              ("library_owns_data", c_uint8)]
class lep_sys_shutter_mode(Structure):
  _fields_ = [("shutterMode", c_uint32),
              ("tempLockoutState", c_uint32),
              ("videoFreezeDuringFFC", c_uint32),
              ("ffcDesired", c_uint32),
              ("elapsedTimeSinceLastFfc", c_uint32),
              ("desiredFfcPeriod", c_uint32),
              ("explicitCmdToOpen", c_bool),
              ("desiredFfcTempDelta", c_uint16),
              ("imminentDelay", c_uint16)]
explicitCmdToOpenVal = 1
desiredFfcTempDeltaVal = 0
imminentDelayVal = 150

sysShutterManual = lep_sys_shutter_mode(0, 0, 1, 0, 0, 180000, explicitCmdToOpenVal, desiredFfcTempDeltaVal, imminentDelayVal)
sysShutterAuto = lep_sys_shutter_mode(1, 0, 1, 0, 0, 180000, explicitCmdToOpenVal, desiredFfcTempDeltaVal, imminentDelayVal)
sysShutterExternal = lep_sys_shutter_mode(2, 0, 1, 0, 0, 180000, explicitCmdToOpenVal, desiredFfcTempDeltaVal, imminentDelayVal)  
  
shutter = lep_sys_shutter_mode()


def print_shutter_info(devh):
    getSDK = 0x3C
    controlID = (getSDK >> 2) + 1
    call_extension_unit(devh, SYS_UNIT_ID, controlID, byref(shutter), 32)
    print("Shutter Info:\n {0}\t shutterMode\n {1}\t tempLockoutState\n {2}\t videoFreezeDuringFFC\n\
 {3}\t ffcDesired\n {4}\t elapsedTimeSinceLastFfc\n {5}\t desiredFfcPeriod\n\
 {6}\t explicitCmdToOpen\n {7}\t desiredFfcTempDelta\n {8}\t imminentDelay\n".format(
        shutter.shutterMode, shutter.tempLockoutState, shutter.videoFreezeDuringFFC,
        shutter.ffcDesired, shutter.elapsedTimeSinceLastFfc, shutter.desiredFfcPeriod,
        shutter.explicitCmdToOpen, shutter.desiredFfcTempDelta, shutter.imminentDelay,
    ))



def call_extension_unit(devh, unit, control, data, size):
  return libuvc.uvc_get_ctrl(devh, unit, control, data, size, 0x81)
def set_extension_unit(devh, unit, control, data, size):
  return libuvc.uvc_set_ctrl(devh, unit, control, data, size, 0x81)

def set_auto_ffc(devh):
    sizeData = 32
    shutter_mode = (c_uint16)(1)
    getSDK = 0x3D
    controlID = (getSDK >> 2) + 1 #formula from Kurt Kiefer
    print('controlID: ' + str(controlID))
    set_extension_unit(devh, SYS_UNIT_ID, controlID, byref(sysShutterAuto), sizeData)

def set_gain_high(devh):
    sizeData = 4
    gain_mode = (c_uint16)(0) #0=HIGH, 1=LOW, 2=AUTO
    setGainSDK = 0x49
    controlID = (setGainSDK >> 2) + 1 #formula from Kurt Kiefer
    print('controlID: ' + str(controlID))
    set_extension_unit(devh, SYS_UNIT_ID, controlID, byref(gain_mode), sizeData) #set_extension_unit(devh, unit, control, data, size)
    perform_manual_ffc(devh)
def perform_manual_ffc(devh):
    sizeData = 1
    shutter_mode = create_string_buffer(sizeData)
    #0x200 Module ID VID
    #0x3C get
    #0x3D set
    getSDK = 0x3D
    runFFC = 0x42
    controlID = (runFFC >> 2) + 1 #formula from Kurt Kiefer
    print('controlID: ' + str(controlID))
    set_extension_unit(devh, SYS_UNIT_ID, controlID, shutter_mode, sizeData) #set_extension_unit(devh, unit, control, data, size)

def ktof(val):
    return round(((1.8 * ktoc(val) + 32.0)), 2)

def ktoc(val):
    return round(((val - 27315) / 100.0), 2)

def display_temperatureF(img, val_k, loc, color):
    val = ktof(val_k)
    cv2.putText(img,"{0:.1f} degF".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    x, y = loc
    cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
    cv2.line(img, (x, y - 2), (x, y + 2), color, 1)

def display_temperatureC(img, val_k, loc, color):
    val = ktoc(val_k)
    cv2.putText(img,"{0:.1f} degC".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    x, y = loc
    cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
    cv2.line(img, (x, y - 2), (x, y + 2), color, 1)

def py_frame_callback(frame, userptr):
    array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
    data = np.frombuffer(
        array_pointer.contents, dtype=np.dtype(np.uint16)).reshape(frame.contents.height, frame.contents.width)
    if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
        return
    if not q.full():
        q.put(data)
       
PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)
#print(PTR_PY_FRAME_CALLBACK)

camState = 'not_recording'
tiff_frame = 1
maxVal = 0
minVal = 0
threshold = 36.0   
fileNum = 1




def getFrame():
    
    global tiff_frame
    global camState
    global maxVal
    global minVal
    data = q.get(True, 500)
    print(data)
    if data is None:
        print('No Data')

    
    #Cannot you cv2.resize on raspberry pi 3b+. Not enough processing power.
    data = cv2.resize(data[:,:], (640, 480))
    
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data)
    img = cv2.LUT(raw_to_8bit(data), generate_colour_map())
    #display_temperature only works if cv2.resize is used
    #display_temperatureC(img, minVal, minLoc, (255, 0, 0)) #displays min temp at min temp location on image
    #print("getFrame")
    if maxLoc[0] > 600:
        # Move text left to keep it on screen
        maxLoc = (maxLoc[0] - 40, maxLoc[1])
    if maxLoc[1] < 20:
        # Move text down to keep it on screen
        maxLoc = (maxLoc[0], 20)
    #if ktoc(maxVal) < threshold:
    #    display_temperatureC(img, maxVal, maxLoc, (0, 255, 0)) #displays max temp at max temp location on image
    else:
        display_temperatureC(img, maxVal, maxLoc, (0, 0, 255)) #displays max temp at max temp location on image
    #display_temperatureF(img, minVal, (10,55), (255, 0, 0)) #display in top left corner the min temp
    
    #print("getFrame")
    #display_temperatureF(img, maxVal, (10,25), (0, 0, 255)) #display in top left corner the max temp
    return img    
    
def raw_to_8bit(data):
    cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
    np.right_shift(data, 8, data)
    return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)    
    

def generate_colour_map():
    """
    Conversion of the colour map from GetThermal to a numpy LUT:
        https://github.com/groupgets/GetThermal/blob/bb467924750a686cc3930f7e3a253818b755a2c0/src/dataformatter.cpp#L6
    """

    lut = np.zeros((256, 1, 3), dtype=np.uint8)

    #colorMaps
    colormap_grayscale = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68, 69, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101, 102, 102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109, 109, 110, 110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 136, 136, 136, 137, 137, 137, 138, 138, 138, 139, 139, 139, 140, 140, 140, 141, 141, 141, 142, 142, 142, 143, 143, 143, 144, 144, 144, 145, 145, 145, 146, 146, 146, 147, 147, 147, 148, 148, 148, 149, 149, 149, 150, 150, 150, 151, 151, 151, 152, 152, 152, 153, 153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156, 157, 157, 157, 158, 158, 158, 159, 159, 159, 160, 160, 160, 161, 161, 161, 162, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165, 165, 166, 166, 166, 167, 167, 167, 168, 168, 168, 169, 169, 169, 170, 170, 170, 171, 171, 171, 172, 172, 172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176, 177, 177, 177, 178, 178, 178, 179, 179, 179, 180, 180, 180, 181, 181, 181, 182, 182, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185, 186, 186, 186, 187, 187, 187, 188, 188, 188, 189, 189, 189, 190, 190, 190, 191, 191, 191, 192, 192, 192, 193, 193, 193, 194, 194, 194, 195, 195, 195, 196, 196, 196, 197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 200, 200, 201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204, 205, 205, 205, 206, 206, 206, 207, 207, 207, 208, 208, 208, 209, 209, 209, 210, 210, 210, 211, 211, 211, 212, 212, 212, 213, 213, 213, 214, 214, 214, 215, 215, 215, 216, 216, 216, 217, 217, 217, 218, 218, 218, 219, 219, 219, 220, 220, 220, 221, 221, 221, 222, 222, 222, 223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226, 227, 227, 227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 231, 231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235, 235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241, 242, 242, 242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 251, 252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255];

    colormap_rainbow = [1, 3, 74, 0, 3, 74, 0, 3, 75, 0, 3, 75, 0, 3, 76, 0, 3, 76, 0, 3, 77, 0, 3, 79, 0, 3, 82, 0, 5, 85, 0, 7, 88, 0, 10, 91, 0, 14, 94, 0, 19, 98, 0, 22, 100, 0, 25, 103, 0, 28, 106, 0, 32, 109, 0, 35, 112, 0, 38, 116, 0, 40, 119, 0, 42, 123, 0, 45, 128, 0, 49, 133, 0, 50, 134, 0, 51, 136, 0, 52, 137, 0, 53, 139, 0, 54, 142, 0, 55, 144, 0, 56, 145, 0, 58, 149, 0, 61, 154, 0, 63, 156, 0, 65, 159, 0, 66, 161, 0, 68, 164, 0, 69, 167, 0, 71, 170, 0, 73, 174, 0, 75, 179, 0, 76, 181, 0, 78, 184, 0, 79, 187, 0, 80, 188, 0, 81, 190, 0, 84, 194, 0, 87, 198, 0, 88, 200, 0, 90, 203, 0, 92, 205, 0, 94, 207, 0, 94, 208, 0, 95, 209, 0, 96, 210, 0, 97, 211, 0, 99, 214, 0, 102, 217, 0, 103, 218, 0, 104, 219, 0, 105, 220, 0, 107, 221, 0, 109, 223, 0, 111, 223, 0, 113, 223, 0, 115, 222, 0, 117, 221, 0, 118, 220, 1, 120, 219, 1, 122, 217, 2, 124, 216, 2, 126, 214, 3, 129, 212, 3, 131, 207, 4, 132, 205, 4, 133, 202, 4, 134, 197, 5, 136, 192, 6, 138, 185, 7, 141, 178, 8, 142, 172, 10, 144, 166, 10, 144, 162, 11, 145, 158, 12, 146, 153, 13, 147, 149, 15, 149, 140, 17, 151, 132, 22, 153, 120, 25, 154, 115, 28, 156, 109, 34, 158, 101, 40, 160, 94, 45, 162, 86, 51, 164, 79, 59, 167, 69, 67, 171, 60, 72, 173, 54, 78, 175, 48, 83, 177, 43, 89, 179, 39, 93, 181, 35, 98, 183, 31, 105, 185, 26, 109, 187, 23, 113, 188, 21, 118, 189, 19, 123, 191, 17, 128, 193, 14, 134, 195, 12, 138, 196, 10, 142, 197, 8, 146, 198, 6, 151, 200, 5, 155, 201, 4, 160, 203, 3, 164, 204, 2, 169, 205, 2, 173, 206, 1, 175, 207, 1, 178, 207, 1, 184, 208, 0, 190, 210, 0, 193, 211, 0, 196, 212, 0, 199, 212, 0, 202, 213, 1, 207, 214, 2, 212, 215, 3, 215, 214, 3, 218, 214, 3, 220, 213, 3, 222, 213, 4, 224, 212, 4, 225, 212, 5, 226, 212, 5, 229, 211, 5, 232, 211, 6, 232, 211, 6, 233, 211, 6, 234, 210, 6, 235, 210, 7, 236, 209, 7, 237, 208, 8, 239, 206, 8, 241, 204, 9, 242, 203, 9, 244, 202, 10, 244, 201, 10, 245, 200, 10, 245, 199, 11, 246, 198, 11, 247, 197, 12, 248, 194, 13, 249, 191, 14, 250, 189, 14, 251, 187, 15, 251, 185, 16, 252, 183, 17, 252, 178, 18, 253, 174, 19, 253, 171, 19, 254, 168, 20, 254, 165, 21, 254, 164, 21, 255, 163, 22, 255, 161, 22, 255, 159, 23, 255, 157, 23, 255, 155, 24, 255, 149, 25, 255, 143, 27, 255, 139, 28, 255, 135, 30, 255, 131, 31, 255, 127, 32, 255, 118, 34, 255, 110, 36, 255, 104, 37, 255, 101, 38, 255, 99, 39, 255, 93, 40, 255, 88, 42, 254, 82, 43, 254, 77, 45, 254, 69, 47, 254, 62, 49, 253, 57, 50, 253, 53, 52, 252, 49, 53, 252, 45, 55, 251, 39, 57, 251, 33, 59, 251, 32, 60, 251, 31, 60, 251, 30, 61, 251, 29, 61, 251, 28, 62, 250, 27, 63, 250, 27, 65, 249, 26, 66, 249, 26, 68, 248, 25, 70, 248, 24, 73, 247, 24, 75, 247, 25, 77, 247, 25, 79, 247, 26, 81, 247, 32, 83, 247, 35, 85, 247, 38, 86, 247, 42, 88, 247, 46, 90, 247, 50, 92, 248, 55, 94, 248, 59, 96, 248, 64, 98, 248, 72, 101, 249, 81, 104, 249, 87, 106, 250, 93, 108, 250, 95, 109, 250, 98, 110, 250, 100, 111, 251, 101, 112, 251, 102, 113, 251, 109, 117, 252, 116, 121, 252, 121, 123, 253, 126, 126, 253, 130, 128, 254, 135, 131, 254, 139, 133, 254, 144, 136, 254, 151, 140, 255, 158, 144, 255, 163, 146, 255, 168, 149, 255, 173, 152, 255, 176, 153, 255, 178, 155, 255, 184, 160, 255, 191, 165, 255, 195, 168, 255, 199, 172, 255, 203, 175, 255, 207, 179, 255, 211, 182, 255, 216, 185, 255, 218, 190, 255, 220, 196, 255, 222, 200, 255, 225, 202, 255, 227, 204, 255, 230, 206, 255, 233, 208]

    colourmap_ironblack = [
        255, 255, 255, 253, 253, 253, 251, 251, 251, 249, 249, 249, 247, 247,
        247, 245, 245, 245, 243, 243, 243, 241, 241, 241, 239, 239, 239, 237,
        237, 237, 235, 235, 235, 233, 233, 233, 231, 231, 231, 229, 229, 229,
        227, 227, 227, 225, 225, 225, 223, 223, 223, 221, 221, 221, 219, 219,
        219, 217, 217, 217, 215, 215, 215, 213, 213, 213, 211, 211, 211, 209,
        209, 209, 207, 207, 207, 205, 205, 205, 203, 203, 203, 201, 201, 201,
        199, 199, 199, 197, 197, 197, 195, 195, 195, 193, 193, 193, 191, 191,
        191, 189, 189, 189, 187, 187, 187, 185, 185, 185, 183, 183, 183, 181,
        181, 181, 179, 179, 179, 177, 177, 177, 175, 175, 175, 173, 173, 173,
        171, 171, 171, 169, 169, 169, 167, 167, 167, 165, 165, 165, 163, 163,
        163, 161, 161, 161, 159, 159, 159, 157, 157, 157, 155, 155, 155, 153,
        153, 153, 151, 151, 151, 149, 149, 149, 147, 147, 147, 145, 145, 145,
        143, 143, 143, 141, 141, 141, 139, 139, 139, 137, 137, 137, 135, 135,
        135, 133, 133, 133, 131, 131, 131, 129, 129, 129, 126, 126, 126, 124,
        124, 124, 122, 122, 122, 120, 120, 120, 118, 118, 118, 116, 116, 116,
        114, 114, 114, 112, 112, 112, 110, 110, 110, 108, 108, 108, 106, 106,
        106, 104, 104, 104, 102, 102, 102, 100, 100, 100, 98, 98, 98, 96, 96,
        96, 94, 94, 94, 92, 92, 92, 90, 90, 90, 88, 88, 88, 86, 86, 86, 84, 84,
        84, 82, 82, 82, 80, 80, 80, 78, 78, 78, 76, 76, 76, 74, 74, 74, 72, 72,
        72, 70, 70, 70, 68, 68, 68, 66, 66, 66, 64, 64, 64, 62, 62, 62, 60, 60,
        60, 58, 58, 58, 56, 56, 56, 54, 54, 54, 52, 52, 52, 50, 50, 50, 48, 48,
        48, 46, 46, 46, 44, 44, 44, 42, 42, 42, 40, 40, 40, 38, 38, 38, 36, 36,
        36, 34, 34, 34, 32, 32, 32, 30, 30, 30, 28, 28, 28, 26, 26, 26, 24, 24,
        24, 22, 22, 22, 20, 20, 20, 18, 18, 18, 16, 16, 16, 14, 14, 14, 12, 12,
        12, 10, 10, 10, 8, 8, 8, 6, 6, 6, 4, 4, 4, 2, 2, 2, 0, 0, 0, 0, 0, 9,
        2, 0, 16, 4, 0, 24, 6, 0, 31, 8, 0, 38, 10, 0, 45, 12, 0, 53, 14, 0,
        60, 17, 0, 67, 19, 0, 74, 21, 0, 82, 23, 0, 89, 25, 0, 96, 27, 0, 103,
        29, 0, 111, 31, 0, 118, 36, 0, 120, 41, 0, 121, 46, 0, 122, 51, 0, 123,
        56, 0, 124, 61, 0, 125, 66, 0, 126, 71, 0, 127, 76, 1, 128, 81, 1, 129,
        86, 1, 130, 91, 1, 131, 96, 1, 132, 101, 1, 133, 106, 1, 134, 111, 1,
        135, 116, 1, 136, 121, 1, 136, 125, 2, 137, 130, 2, 137, 135, 3, 137,
        139, 3, 138, 144, 3, 138, 149, 4, 138, 153, 4, 139, 158, 5, 139, 163,
        5, 139, 167, 5, 140, 172, 6, 140, 177, 6, 140, 181, 7, 141, 186, 7,
        141, 189, 10, 137, 191, 13, 132, 194, 16, 127, 196, 19, 121, 198, 22,
        116, 200, 25, 111, 203, 28, 106, 205, 31, 101, 207, 34, 95, 209, 37,
        90, 212, 40, 85, 214, 43, 80, 216, 46, 75, 218, 49, 69, 221, 52, 64,
        223, 55, 59, 224, 57, 49, 225, 60, 47, 226, 64, 44, 227, 67, 42, 228,
        71, 39, 229, 74, 37, 230, 78, 34, 231, 81, 32, 231, 85, 29, 232, 88,
        27, 233, 92, 24, 234, 95, 22, 235, 99, 19, 236, 102, 17, 237, 106, 14,
        238, 109, 12, 239, 112, 12, 240, 116, 12, 240, 119, 12, 241, 123, 12,
        241, 127, 12, 242, 130, 12, 242, 134, 12, 243, 138, 12, 243, 141, 13,
        244, 145, 13, 244, 149, 13, 245, 152, 13, 245, 156, 13, 246, 160, 13,
        246, 163, 13, 247, 167, 13, 247, 171, 13, 248, 175, 14, 248, 178, 15,
        249, 182, 16, 249, 185, 18, 250, 189, 19, 250, 192, 20, 251, 196, 21,
        251, 199, 22, 252, 203, 23, 252, 206, 24, 253, 210, 25, 253, 213, 27,
        254, 217, 28, 254, 220, 29, 255, 224, 30, 255, 227, 39, 255, 229, 53,
        255, 231, 67, 255, 233, 81, 255, 234, 95, 255, 236, 109, 255, 238, 123,
        255, 240, 137, 255, 242, 151, 255, 244, 165, 255, 246, 179, 255, 248,
        193, 255, 249, 207, 255, 251, 221, 255, 253, 235, 255, 255, 24]

    def chunk(ulist, step):
        return map(lambda i: ulist[i: i + step], range(0, len(ulist), step))

    if (colorMapType == 1):
        chunks = chunk(colormap_rainbow, 3)
    elif (colorMapType == 2):
        chunks = chunk(colormap_grayscale, 3)
    else:
        chunks = chunk(colourmap_ironblack, 3)

    red = []
    green = []
    blue = []

    for chunk in chunks:
        red.append(chunk[0])
        green.append(chunk[1])
        blue.append(chunk[2])

    lut[:, 0, 0] = blue

    lut[:, 0, 1] = green

    lut[:, 0, 2] = red

    return lut 
    

    
    
    
PT_USB_VID = 0x1e4e
PT_USB_PID = 0x0100

AGC_UNIT_ID = 3
OEM_UNIT_ID = 4
RAD_UNIT_ID = 5
SYS_UNIT_ID = 6
VID_UNIT_ID = 7

UVC_FRAME_FORMAT_UYVY = 4
UVC_FRAME_FORMAT_I420 = 5
UVC_FRAME_FORMAT_RGB = 7
UVC_FRAME_FORMAT_BGR = 8
UVC_FRAME_FORMAT_Y16 = 13
libuvc.uvc_get_format_descs.restype = POINTER(uvc_format_desc)


VS_FMT_GUID_GREY = create_string_buffer(
    b"Y8  \x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71", 16
)

VS_FMT_GUID_Y16 = create_string_buffer(
    b"Y16 \x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71", 16
)


  
def startStream():
    global devh
    ctx = POINTER(uvc_context)()
    dev = POINTER(uvc_device)()
    devh = POINTER(uvc_device_handle)()
    ctrl = uvc_stream_ctrl()
    res = libuvc.uvc_init(byref(ctx), 0)

    if res < 0:
        print("uvc_init error")
        exit(1)
                
    try:
        res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
        if res < 0:
            print("uvc_find_device error")
            exit(1)

        try:
            res = libuvc.uvc_open(dev, ctypes.byref(devh))
            #print(res)
            if res < 0:
                print("uvc_open error")
                exit(1)
        
            print("device opened!")
            print_device_info(devh)
            #print("hi")
            print_device_formats(devh)
            #print("hi")
            frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
        
    
    
            if len(frame_formats) == 0:
                print("device does not support Y16")
                exit(1)
    
            libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
                                               frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval)
                                               )
    
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            #data = q.get()
            #print(data)
            #print("hi")
            res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
            #print("hi")
            if res < 0:
                print("uvc_start_streaming failed: {0}".format(res))
                exit(1)
            
            print("done starting stream, displaying settings")
            
            print_shutter_info(devh)
            print("resetting settings to default")
            
            set_auto_ffc(devh)
            set_gain_high(devh)
            print("current settings")
            print_shutter_info(devh)
        except:
            #libuvc.uvc_unref_device(dev)
            print('Failed to Open Device')
    except:
        #libuvc.uvc_exit(ctx)
        print('Failed to Find Device')
        exit(1)



class MyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        

    def run(self):
        print('Start Stream')
        while True:
            
            frame = getFrame()
            #print('while')
            #print(frame)
            #rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #cv2.imshow("이미지", rgbImage)  # 이미지 표시
            #cv2.waitKey(1)  # 표시를 갱신하기 위해 잠시 대기
           # convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
            #p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
            #self.changePixmap(p)

def startThread():
    global thread
    try:
        if thread == "unactive":
            try:
                startStream()
                
                thread = MyThread()  # 이미지 변경 콜백 함수 설정
                #print("hi")
                thread.start()  # 스레드 시작
                print('Starting Thread')
            except:
                print('Failed!!!!')
                exit(1)
        else:
            print('Already Started Camera')
    except:
        print('Error Starting Camera - Plug or Re-Plug Camera into Computer, Wait at Least 10 Seconds, then Try Again.')
thread = "unactive"
startThread()  
