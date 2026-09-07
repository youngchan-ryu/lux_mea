"""Helios USB diagnosis that stops before the call that panics the kernel.

macOS 26.6.2 (25G83) panics inside IOUSBHostFamily when libusb reads pipe
properties for the Helios interface's alternate setting 1:

    OpenDevices -> HeliosDac::_OpenUsbDevices
                -> libusb_set_interface_alt_setting(h, 0, 1)
                -> darwin_set_interface_altsetting
                -> get_endpoints
                -> darwin_get_pipe_properties +112  <- IOUSBLib vtable +0x1e8
                -> [kernel] IOUSBHostFamily         <- read of 0x3, panic

The +0x1e8 slot is a descriptor-property call that libusb only started making
after 1.0.26; the +0x1e0 call right before it returns fine. libusb 1.0.26 never
touches +0x1e8 anywhere in the binary, which is why dropping the 1.0.26 built by
Grix (helios_dac/sdk/cpp/shared_library/xcode/helios_driver/) next to
libHeliosLaserDAC.dylib is the fix candidate -- its LC_RPATH is @loader_path, so
a sibling libusb wins over the Homebrew one with no system change.

Everything before that call is proven safe: the panic log shows those frames
already returned. So this script runs init / enumerate / read descriptors /
open / claim_interface and then STOPS. It cannot take the machine down.

    python3 usbprobe.py

The alt-setting call is behind an explicit opt-in. Only pass it after changing
something (cable, hub, port, libusb version), and save your work first:

    python3 usbprobe.py --danger-altset

Pure stdlib ctypes -- no venv needed, and it never loads libHeliosLaserDAC.
"""
import argparse
import ctypes
import ctypes.util
import os
import sys

HELIOS_VID = 0x1209
HELIOS_PID = 0xE500

HERE = os.path.dirname(os.path.abspath(__file__))

# Same order dyld uses for libHeliosLaserDAC.dylib: its LC_RPATH is @loader_path,
# so a libusb sitting next to it wins over the Homebrew one. Probing in this
# order means we test exactly what firstlight.py will end up loading.
LIBUSB_CANDIDATES = (
    os.path.join(HERE, "libusb-1.0.0.dylib"),
    "/opt/homebrew/lib/libusb-1.0.0.dylib",
    "/usr/local/lib/libusb-1.0.0.dylib",
    "/opt/homebrew/lib/libusb-1.0.dylib",
)

SPEED = {0: "unknown", 1: "low (1.5 Mbit/s)", 2: "full (12 Mbit/s)",
         3: "high (480 Mbit/s)", 4: "super (5 Gbit/s)", 5: "super+ (10 Gbit/s)"}
XFER = {0: "control", 1: "isochronous", 2: "bulk", 3: "interrupt"}


class DeviceDescriptor(ctypes.Structure):
    _fields_ = [("bLength", ctypes.c_uint8), ("bDescriptorType", ctypes.c_uint8),
                ("bcdUSB", ctypes.c_uint16), ("bDeviceClass", ctypes.c_uint8),
                ("bDeviceSubClass", ctypes.c_uint8), ("bDeviceProtocol", ctypes.c_uint8),
                ("bMaxPacketSize0", ctypes.c_uint8), ("idVendor", ctypes.c_uint16),
                ("idProduct", ctypes.c_uint16), ("bcdDevice", ctypes.c_uint16),
                ("iManufacturer", ctypes.c_uint8), ("iProduct", ctypes.c_uint8),
                ("iSerialNumber", ctypes.c_uint8), ("bNumConfigurations", ctypes.c_uint8)]


class EndpointDescriptor(ctypes.Structure):
    _fields_ = [("bLength", ctypes.c_uint8), ("bDescriptorType", ctypes.c_uint8),
                ("bEndpointAddress", ctypes.c_uint8), ("bmAttributes", ctypes.c_uint8),
                ("wMaxPacketSize", ctypes.c_uint16), ("bInterval", ctypes.c_uint8),
                ("bRefresh", ctypes.c_uint8), ("bSynchAddress", ctypes.c_uint8),
                ("extra", ctypes.POINTER(ctypes.c_ubyte)), ("extra_length", ctypes.c_int)]


class InterfaceDescriptor(ctypes.Structure):
    _fields_ = [("bLength", ctypes.c_uint8), ("bDescriptorType", ctypes.c_uint8),
                ("bInterfaceNumber", ctypes.c_uint8), ("bAlternateSetting", ctypes.c_uint8),
                ("bNumEndpoints", ctypes.c_uint8), ("bInterfaceClass", ctypes.c_uint8),
                ("bInterfaceSubClass", ctypes.c_uint8), ("bInterfaceProtocol", ctypes.c_uint8),
                ("iInterface", ctypes.c_uint8),
                ("endpoint", ctypes.POINTER(EndpointDescriptor)),
                ("extra", ctypes.POINTER(ctypes.c_ubyte)), ("extra_length", ctypes.c_int)]


class Interface(ctypes.Structure):
    _fields_ = [("altsetting", ctypes.POINTER(InterfaceDescriptor)),
                ("num_altsetting", ctypes.c_int)]


class ConfigDescriptor(ctypes.Structure):
    _fields_ = [("bLength", ctypes.c_uint8), ("bDescriptorType", ctypes.c_uint8),
                ("wTotalLength", ctypes.c_uint16), ("bNumInterfaces", ctypes.c_uint8),
                ("bConfigurationValue", ctypes.c_uint8), ("iConfiguration", ctypes.c_uint8),
                ("bmAttributes", ctypes.c_uint8), ("MaxPower", ctypes.c_uint8),
                ("interface", ctypes.POINTER(Interface)),
                ("extra", ctypes.POINTER(ctypes.c_ubyte)), ("extra_length", ctypes.c_int)]


class Version(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint16), ("minor", ctypes.c_uint16),
                ("micro", ctypes.c_uint16), ("nano", ctypes.c_uint16),
                ("rc", ctypes.c_char_p), ("describe", ctypes.c_char_p)]


def load_libusb():
    """Resolve the same libusb that libHeliosLaserDAC.dylib picks up."""
    env = os.environ.get("LIBUSB_PATH")
    paths = ([env] if env else []) + list(LIBUSB_CANDIDATES)
    found = ctypes.util.find_library("usb-1.0")
    if found:
        paths.append(found)
    for p in paths:
        if p and (os.path.exists(p) or p == found):
            try:
                return ctypes.CDLL(p), p
            except OSError:
                continue
    sys.exit("[!] libusb-1.0 을 찾지 못함. LIBUSB_PATH=/경로/libusb-1.0.0.dylib 로 지정.")


def bind(usb):
    usb.libusb_get_version.restype = ctypes.POINTER(Version)
    usb.libusb_init.argtypes = [ctypes.c_void_p]
    usb.libusb_get_device_list.argtypes = [ctypes.c_void_p,
                                           ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))]
    usb.libusb_get_device_list.restype = ctypes.c_ssize_t
    usb.libusb_get_device_descriptor.argtypes = [ctypes.c_void_p,
                                                 ctypes.POINTER(DeviceDescriptor)]
    usb.libusb_get_config_descriptor.argtypes = [
        ctypes.c_void_p, ctypes.c_uint8, ctypes.POINTER(ctypes.POINTER(ConfigDescriptor))]
    usb.libusb_free_config_descriptor.argtypes = [ctypes.POINTER(ConfigDescriptor)]
    usb.libusb_get_bus_number.argtypes = [ctypes.c_void_p]
    usb.libusb_get_bus_number.restype = ctypes.c_uint8
    usb.libusb_get_device_address.argtypes = [ctypes.c_void_p]
    usb.libusb_get_device_address.restype = ctypes.c_uint8
    usb.libusb_get_device_speed.argtypes = [ctypes.c_void_p]
    usb.libusb_get_port_numbers.argtypes = [ctypes.c_void_p,
                                            ctypes.POINTER(ctypes.c_uint8), ctypes.c_int]
    usb.libusb_open.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    usb.libusb_close.argtypes = [ctypes.c_void_p]
    usb.libusb_claim_interface.argtypes = [ctypes.c_void_p, ctypes.c_int]
    usb.libusb_release_interface.argtypes = [ctypes.c_void_p, ctypes.c_int]
    usb.libusb_set_interface_alt_setting.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    usb.libusb_error_name.argtypes = [ctypes.c_int]
    usb.libusb_error_name.restype = ctypes.c_char_p
    usb.libusb_free_device_list.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]


def err(usb, code):
    return f"{code} ({usb.libusb_error_name(code).decode()})"


def dump_config(usb, dev):
    """Parse the cached configuration descriptor.

    The darwin backend caches this at enumeration time, so it is read out of
    process memory -- no pipe properties, no kernel round trip, no panic.
    """
    cfg = ctypes.POINTER(ConfigDescriptor)()
    rc = usb.libusb_get_config_descriptor(dev, 0, ctypes.byref(cfg))
    if rc != 0:
        print(f"  [!] config descriptor 읽기 실패: {err(usb, rc)}")
        return
    c = cfg.contents
    print(f"  configuration {c.bConfigurationValue}: "
          f"인터페이스 {c.bNumInterfaces}개, wTotalLength {c.wTotalLength}, "
          f"MaxPower {c.MaxPower * 2} mA")
    for i in range(c.bNumInterfaces):
        iface = c.interface[i]
        for a in range(iface.num_altsetting):
            alt = iface.altsetting[a]
            print(f"    interface {alt.bInterfaceNumber} alt {alt.bAlternateSetting}: "
                  f"엔드포인트 {alt.bNumEndpoints}개  "
                  f"(class {alt.bInterfaceClass:#04x}/{alt.bInterfaceSubClass:#04x})")
            for e in range(alt.bNumEndpoints):
                ep = alt.endpoint[e]
                kind = XFER.get(ep.bmAttributes & 0x03, "?")
                direction = "IN " if ep.bEndpointAddress & 0x80 else "OUT"
                warn = ""
                if kind == "isochronous":
                    warn = "   <-- Helios 는 안 쓰지만 커널이 속성을 읽다 죽는다"
                print(f"      ep {ep.bEndpointAddress:#04x} {direction} "
                      f"{kind:<12} wMaxPacketSize={ep.wMaxPacketSize:<4} "
                      f"bInterval={ep.bInterval}{warn}")
    usb.libusb_free_config_descriptor(cfg)


def port_path(usb, dev):
    buf = (ctypes.c_uint8 * 8)()
    n = usb.libusb_get_port_numbers(dev, buf, 8)
    if n <= 0:
        return "?"
    return "-".join(str(buf[i]) for i in range(n))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--danger-altset", action="store_true",
                    help="alt setting 1 로 전환까지 시도한다. 커널 패닉으로 맥이 즉시 "
                         "꺼질 수 있다. 작업을 저장/커밋한 뒤에만 사용할 것.")
    a = ap.parse_args()

    usb, path = load_libusb()
    bind(usb)
    v = usb.libusb_get_version().contents
    print(f"[i] libusb {v.major}.{v.minor}.{v.micro}.{v.nano} "
          f"({v.describe.decode() if v.describe else '-'})")
    print(f"[i] 경로: {path}")

    rc = usb.libusb_init(None)
    if rc != 0:
        sys.exit(f"[!] libusb_init 실패: {err(usb, rc)}")
    print("[1/5] libusb_init OK")

    devs = ctypes.POINTER(ctypes.c_void_p)()
    cnt = usb.libusb_get_device_list(None, ctypes.byref(devs))
    if cnt < 0:
        sys.exit(f"[!] 장치 목록 실패: {err(usb, cnt)}")
    print(f"[2/5] 버스에 USB 장치 {cnt}개")

    target, desc = None, None
    for i in range(cnt):
        d = DeviceDescriptor()
        if usb.libusb_get_device_descriptor(devs[i], ctypes.byref(d)) < 0:
            continue
        if d.idVendor == HELIOS_VID and d.idProduct == HELIOS_PID:
            target, desc = devs[i], d
            break

    if target is None:
        usb.libusb_free_device_list(devs, 1)
        usb.libusb_exit(None)
        sys.exit(f"[!] Helios({HELIOS_VID:#06x}:{HELIOS_PID:#06x}) 를 못 찾음. "
                 "전원/케이블 확인.")

    speed = usb.libusb_get_device_speed(target)
    print(f"[3/5] Helios 발견")
    print(f"  bus {usb.libusb_get_bus_number(target)} "
          f"addr {usb.libusb_get_device_address(target)} "
          f"port path {port_path(usb, target)}")
    print(f"  링크 속도: {SPEED.get(speed, speed)}"
          f"{'   <-- 허브 뒤라면 TT split transaction 경로' if speed == 2 else ''}")
    print(f"  bcdUSB {desc.bcdUSB:#06x}  bMaxPacketSize0 {desc.bMaxPacketSize0}  "
          f"bcdDevice {desc.bcdDevice:#06x}")
    dump_config(usb, target)

    handle = ctypes.c_void_p()
    rc = usb.libusb_open(target, ctypes.byref(handle))
    usb.libusb_free_device_list(devs, 1)
    if rc != 0:
        usb.libusb_exit(None)
        sys.exit(f"[!] libusb_open 실패: {err(usb, rc)}")
    print("[4/5] libusb_open OK")

    rc = usb.libusb_claim_interface(handle, 0)
    if rc != 0:
        usb.libusb_close(handle)
        usb.libusb_exit(None)
        sys.exit(f"[!] claim_interface 실패: {err(usb, rc)}")
    print("[5/5] claim_interface(0) OK  <- alt 0 은 엔드포인트가 없어 파이프 조회가 없다")

    print()
    print("여기까지가 안전 구간이다. 장치, 케이블, 허브, libusb, 디스크립터 모두 정상.")
    print("다음 호출 libusb_set_interface_alt_setting(h, 0, 1) 이 커널을 죽인다.")

    if a.danger_altset:
        print()
        print("!!! --danger-altset 지정됨. 3초 뒤 alt setting 1 로 전환한다.")
        print("!!! 여기서 맥이 꺼지면 조건이 아직 그대로라는 뜻이다. Ctrl+C 로 중단.")
        import time
        for s in (3, 2, 1):
            print(f"!!! {s}...", flush=True)
            time.sleep(1)
        rc = usb.libusb_set_interface_alt_setting(handle, 0, 1)
        if rc == 0:
            print(">>> alt setting 1 성공! 패닉이 해결되었다. firstlight.py 를 돌려도 된다.")
        else:
            print(f">>> alt setting 실패(패닉은 아님): {err(usb, rc)}")

    usb.libusb_release_interface(handle, 0)
    usb.libusb_close(handle)
    usb.libusb_exit(None)
    print("[i] 정리 완료.")


if __name__ == "__main__":
    main()
