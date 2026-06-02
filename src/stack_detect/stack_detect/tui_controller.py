"""
tui_controller.py  –  Robot operation TUI
==========================================

Key bindings
------------
  Slides      →  / L   : next slide          ←  / H   : previous slide
  Modules     1         : UNSTACK             2         : GRASP
              3         : UNFOLD              4         : BAG OPEN
  Reset       R         : reset unfold/bagopen flags + queue
  Quit        Q

  During unstack confirmation modal:
              Y + Enter : confirm / continue
              N + Enter : reject / retry

Slide-gate logic
----------------
  UNFOLD_DONE_SLIDE  is gated on /unfold_done   publishing True.
  BAGOPEN_DONE_SLIDE is gated on /bag_open_done publishing True.

  Flag is STICKY once True – navigating to the gated slide is instant.
  If the module is currently running (triggered but not done) and you press
  → toward the gated slide, the intent is QUEUED and fires automatically
  when the done signal arrives.
  If the module has never been triggered, pressing → toward the gated slide
  warns and blocks.

  R resets both flags to False, clears running state and any queued nav,
  and publishes False on /unfold_done and /bag_open_done so the modules
  know the state has been cleared.

Unstack integration
-------------------
  Pressing 1 instantiates StackDetectorDINO (with_slides=True) and runs
  tui_unstack_utils.run() in a background thread. input() calls inside
  the sequence are routed via _ask(): confirmation prompts show a TUI
  modal; everything else is auto-confirmed. stack_choice is not imported.
"""

import sys
import io
import curses
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from softenable_display_msgs.srv import SetDisplay
from stack_approach.motion_helper_v2 import MotionHelperV2

import tui_unstack_utils
import tui_bagopen_utils


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

N_SLIDES            = 13

UNFOLD_DONE_SLIDE   = 8
BAGOPEN_DONE_SLIDE  = 11

TOPIC_UNFOLD_DONE   = "/unfold_done"
TOPIC_BAGOPEN_DONE  = "/bag_open_done"
SRV_SET_DISPLAY     = "/set_display"
SRV_GRASP           = "/grasp_dual_service"
SRV_UNFOLD          = "/gown_unfold_service"
SRV_BAGOPEN         = "/bag_open/trigger"

SLIDE_PREFIX        = "study"


# ═══════════════════════════════════════════════════════════════════════════
#  Colour pair indices
# ═══════════════════════════════════════════════════════════════════════════

C_TITLE    = 1
C_SLIDE    = 2
C_ROBOT    = 3
C_LOCKED   = 4
C_UNLOCKED = 5
C_LOG      = 6
C_KEY      = 7
C_DIM      = 8
C_WARN     = 9
C_QUEUE    = 10
C_RUNNING  = 11
C_RESET    = 12
C_MODAL    = 13   # bright yellow-on-black – modal overlay


# ═══════════════════════════════════════════════════════════════════════════
#  Unstack confirmation gate
#  The background thread calls tui_ask(prompt) instead of input(prompt).
#  The curses loop feeds answers via UnstackGate.answer(text).
# ═══════════════════════════════════════════════════════════════════════════

class UnstackGate:
    """
    Thread-safe Y/N gate.  The worker thread calls .ask(prompt) and blocks.
    The TUI loop calls .answer(text) when the user presses Enter.
    """
    def __init__(self):
        self._event   = threading.Event()
        self._answer  = ""
        self.prompt   = ""          # set by ask(), read by TUI for display
        self.waiting  = False       # True while thread is blocked in ask()

    def ask(self, prompt: str) -> str:
        """Called from worker thread.  Blocks until TUI provides an answer."""
        self.prompt  = prompt
        self.waiting = True
        self._event.clear()
        self._event.wait()          # released by answer()
        self.waiting = False
        return self._answer

    def answer(self, text: str):
        """Called from curses thread.  Unblocks the worker."""
        self._answer = text
        self._event.set()


# ═══════════════════════════════════════════════════════════════════════════
#  Application state
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AppState:
    current_slide:    int            = 1

    unfold_done:      bool           = False
    bagopen_done:     bool           = False

    unfold_running:   bool           = False
    bagopen_running:  bool           = False

    queued_slide:     Optional[int]  = None

    # Unstack
    unstack_running:  bool           = False
    unstack_gate:     UnstackGate    = field(default_factory=UnstackGate)
    # Bag open
    bagopen_gate:     UnstackGate    = field(default_factory=UnstackGate)
    # Buffer typed by user while modal is open (before Enter)
    # Shared by both unstack and bagopen modals (only one can be active at a time)
    modal_input:      str            = ""

    log_lines:        list           = field(default_factory=list)
    ros_ready:        bool           = False
    last_action:      str            = ""
    last_action_time: float          = 0.0

    _lock:            object         = field(default_factory=threading.Lock,
                                            repr=False, compare=False)

    # ── helpers ──────────────────────────────────────────────────────────

    def in_modal(self) -> bool:
        return ((self.unstack_running and self.unstack_gate.waiting) or
                (self.bagopen_running  and self.bagopen_gate.waiting))

    def active_gate(self):
        """Return whichever gate is currently waiting, or None."""
        if self.unstack_running and self.unstack_gate.waiting:
            return self.unstack_gate
        if self.bagopen_running and self.bagopen_gate.waiting:
            return self.bagopen_gate
        return None

    def gated_slides(self) -> dict:
        result = {}
        if UNFOLD_DONE_SLIDE:
            if self.unfold_done:           result[UNFOLD_DONE_SLIDE] = 'open'
            elif self.unfold_running:      result[UNFOLD_DONE_SLIDE] = 'running'
            else:                          result[UNFOLD_DONE_SLIDE] = 'locked'
        if BAGOPEN_DONE_SLIDE:
            if self.bagopen_done:          result[BAGOPEN_DONE_SLIDE] = 'open'
            elif self.bagopen_running:     result[BAGOPEN_DONE_SLIDE] = 'running'
            else:                          result[BAGOPEN_DONE_SLIDE] = 'locked'
        return result

    def add_log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        with self._lock:
            self.log_lines.append(f"[{ts}]  {msg}")
            if len(self.log_lines) > 200:
                self.log_lines = self.log_lines[-200:]

    def do_reset(self):
        with self._lock:
            self.unfold_done     = False
            self.bagopen_done    = False
            self.unfold_running  = False
            self.bagopen_running = False
            self.queued_slide    = None
            self.current_slide   = 1
        self.add_log("↺ RESET – flags cleared, re-locked, queue cleared, back to slide 1")


# ═══════════════════════════════════════════════════════════════════════════
#  ROS2 node
# ═══════════════════════════════════════════════════════════════════════════

class RobotNode(Node):
    def __init__(self, state: AppState, on_slide_ready):
        super().__init__("tui_controller")
        self.state        = state
        self._slide_ready = on_slide_ready

        self.cli_display = self.create_client(SetDisplay, SRV_SET_DISPLAY)
        self.cli_grasp   = self.create_client(Trigger,    SRV_GRASP)
        self.cli_unfold  = self.create_client(Trigger,    SRV_UNFOLD)
        self.cli_bagopen = self.create_client(Trigger,    SRV_BAGOPEN)

        self.pub_unfold_done  = self.create_publisher(Bool, TOPIC_UNFOLD_DONE,  1)
        self.pub_bagopen_done = self.create_publisher(Bool, TOPIC_BAGOPEN_DONE, 1)

        self.create_subscription(Bool, TOPIC_UNFOLD_DONE,
                                 self._cb_unfold_done,  10)
        self.create_subscription(Bool, TOPIC_BAGOPEN_DONE,
                                 self._cb_bagopen_done, 10)

        state.ros_ready = True
        state.add_log("ROS2 node ready")

    # ── done callbacks ────────────────────────────────────────────────────

    def _cb_unfold_done(self, msg: Bool):
        if not msg.data:
            return
        with self.state._lock:
            if self.state.unfold_done:
                return
            self.state.unfold_done    = True
            self.state.unfold_running = False
            queued = self.state.queued_slide
            if queued == UNFOLD_DONE_SLIDE:
                self.state.queued_slide = None
        self.state.add_log(f"✓ Unfold finished  →  slide {UNFOLD_DONE_SLIDE} unlocked")
        if queued == UNFOLD_DONE_SLIDE:
            self.state.add_log(f"  queued intent fired  →  showing slide {UNFOLD_DONE_SLIDE}")
            self._slide_ready(UNFOLD_DONE_SLIDE)

    def _cb_bagopen_done(self, msg: Bool):
        if not msg.data:
            return
        with self.state._lock:
            if self.state.bagopen_done:
                return
            self.state.bagopen_done    = True
            self.state.bagopen_running = False
            queued = self.state.queued_slide
            if queued == BAGOPEN_DONE_SLIDE:
                self.state.queued_slide = None
        self.state.add_log(f"✓ Bag-open finished  →  slide {BAGOPEN_DONE_SLIDE} unlocked")
        if queued == BAGOPEN_DONE_SLIDE:
            self.state.add_log(f"  queued intent fired  →  showing slide {BAGOPEN_DONE_SLIDE}")
            self._slide_ready(BAGOPEN_DONE_SLIDE)

    # ── service / publish helpers ─────────────────────────────────────────

    def set_display(self, slide_no: int):
        if not self.cli_display.service_is_ready():
            self.state.add_log("⚠  /set_display not available")
            return
        req      = SetDisplay.Request()
        req.name = f"{SLIDE_PREFIX}-{slide_no}"
        req.use_tts = True
        future   = self.cli_display.call_async(req)
        future.add_done_callback(
            lambda f: self.state.add_log(
                f"  slide {slide_no} display → "
                f"{'ok' if f.result() and f.result().success else 'FAIL'}"
            )
        )

    def _trigger(self, client, label: str):
        if not client.service_is_ready():
            self.state.add_log(f"⚠  {label} service not available")
            return
        future = client.call_async(Trigger.Request())
        future.add_done_callback(
            lambda f: self.state.add_log(
                f"  {label} → {'ok' if f.result() and f.result().success else 'FAIL'}"
            )
        )

    def trigger_grasp(self):
        self._trigger(self.cli_grasp, "GRASP")

    def trigger_unfold(self):
        with self.state._lock:
            self.state.unfold_running = True
        self._trigger(self.cli_unfold, "UNFOLD")

    def trigger_bagopen(self):
        with self.state._lock:
            self.state.bagopen_running = True
        self._trigger(self.cli_bagopen, "BAG OPEN")

    def publish_reset(self):
        false_msg      = Bool()
        false_msg.data = False
        self.pub_unfold_done.publish(false_msg)
        self.pub_bagopen_done.publish(false_msg)
        self.state.add_log("  published False → /unfold_done, /bag_open_done")


# ═══════════════════════════════════════════════════════════════════════════
#  Unstack worker  (runs in a daemon thread)
# ═══════════════════════════════════════════════════════════════════════════

# run_unstack logic lives in tui_unstack_utils.run()


# ═══════════════════════════════════════════════════════════════════════════
#  Curses drawing helpers
# ═══════════════════════════════════════════════════════════════════════════

def init_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(C_TITLE,   curses.COLOR_WHITE,   curses.COLOR_BLUE)
    curses.init_pair(C_SLIDE,   curses.COLOR_CYAN,    -1)
    curses.init_pair(C_ROBOT,   curses.COLOR_GREEN,   -1)
    curses.init_pair(C_LOCKED,  curses.COLOR_RED,     -1)
    curses.init_pair(C_UNLOCKED,curses.COLOR_GREEN,   -1)
    curses.init_pair(C_LOG,     curses.COLOR_WHITE,   -1)
    curses.init_pair(C_KEY,     curses.COLOR_BLACK,   curses.COLOR_WHITE)
    curses.init_pair(C_DIM,     curses.COLOR_BLACK,   -1)
    curses.init_pair(C_WARN,    curses.COLOR_YELLOW,  -1)
    curses.init_pair(C_QUEUE,   curses.COLOR_CYAN,    -1)
    curses.init_pair(C_RUNNING, curses.COLOR_MAGENTA, -1)
    curses.init_pair(C_RESET,   curses.COLOR_RED,     curses.COLOR_WHITE)
    curses.init_pair(C_MODAL,   curses.COLOR_YELLOW,  -1)


def _put(win, y, x, text, attr=0):
    try:
        win.addstr(y, x, text, attr)
    except curses.error:
        pass


def draw_badge(win, y, x, key_text: str, label: str, pair: int = C_KEY):
    badge = f" {key_text} "
    _put(win, y, x, badge, curses.color_pair(pair) | curses.A_BOLD)
    _put(win, y, x + len(badge), f"  {label}")
    return x + len(badge) + 2 + len(label)


def draw_slide_bar(win, state: AppState, y: int, width: int):
    gated  = state.gated_slides()
    queued = state.queued_slide
    _put(win, y, 2, "SLIDES", curses.color_pair(C_SLIDE) | curses.A_BOLD)
    col = 10
    for i in range(1, N_SLIDES + 1):
        status     = gated.get(i, 'open')
        is_current = (i == state.current_slide)
        is_queued  = (i == queued)

        if is_current:
            attr   = curses.color_pair(C_KEY) | curses.A_BOLD
            marker = f"[{i:2d}]"
        elif is_queued:
            attr   = curses.color_pair(C_QUEUE) | curses.A_BOLD
            marker = f">{i:2d}<"
        elif status == 'running':
            attr   = curses.color_pair(C_RUNNING)
            marker = f" {i:2d}~"
        elif status == 'locked':
            attr   = curses.color_pair(C_LOCKED)
            marker = f" {i:2d}L"
        else:
            attr   = curses.color_pair(C_SLIDE)
            marker = f" {i:2d} "

        _put(win, y, col, marker, attr)
        col += len(marker) + 1
        if col > width - 5:
            break


def draw_robot_panel(win, state: AppState, y: int, width: int):
    _put(win, y, 2, "ROBOT MODULES", curses.color_pair(C_ROBOT) | curses.A_BOLD)

    r1 = y + 2
    # UNSTACK – dim if already running
    unstack_dim = state.unstack_running
    unstack_pair = C_DIM if unstack_dim else C_KEY
    draw_badge(win, r1, 4,  "1", "UNSTACK", pair=unstack_pair)
    if state.unstack_running:
        _put(win, r1, 18, " (running…)", curses.color_pair(C_RUNNING) | curses.A_BOLD)
    draw_badge(win, r1, 30, "2", "GRASP")

    r2 = y + 4
    draw_badge(win, r2, 4,  "3", "UNFOLD")
    _draw_gate_hint(win, r2, 18, UNFOLD_DONE_SLIDE,
                    state.unfold_done, state.unfold_running)
    draw_badge(win, r2, 30, "4", "BAG OPEN")
    _draw_gate_hint(win, r2, 44, BAGOPEN_DONE_SLIDE,
                    state.bagopen_done, state.bagopen_running)

    # RESET – separated by a line, red badge
    r3 = y + 6
    _put(win, r3, 2, "─" * min(width - 3, 70), curses.color_pair(C_DIM))
    draw_badge(win, r3 + 1, 4, "R",
               "RESET  (clears flags, re-locks slides, drops queue)",
               pair=C_RESET)


def _draw_gate_hint(win, y, x, slide_no, done: bool, running: bool):
    if not slide_no:
        return
    if done:
        _put(win, y, x, f"slide {slide_no}: unlocked",
             curses.color_pair(C_UNLOCKED))
    elif running:
        _put(win, y, x, f"slide {slide_no}: running…",
             curses.color_pair(C_RUNNING) | curses.A_BOLD)
    else:
        _put(win, y, x, f"slide {slide_no}: waiting",
             curses.color_pair(C_LOCKED))


def draw_modal(win, state: AppState, h: int, w: int):
    """
    Overlay shown when unstack worker is waiting for Y/N input.
    Drawn over the centre of the screen.
    """
    gate   = state.active_gate()
    if gate is None:
        return
    prompt = gate.prompt.strip() or "continue?"
    title  = " UNSTACK CONFIRMATION " if state.unstack_gate.waiting else " BAG OPEN CONFIRMATION "

    box_w  = min(w - 6, 60)
    box_h  = 7
    box_y  = (h - box_h) // 2
    box_x  = (w - box_w) // 2

    attr_box   = curses.color_pair(C_MODAL) | curses.A_BOLD
    attr_inner = curses.color_pair(C_MODAL)

    # Border
    for row in range(box_h):
        _put(win, box_y + row, box_x, " " * box_w, attr_box)

    _put(win, box_y,             box_x + 2, "┌" + "─" * (box_w - 4) + "┐", attr_box)
    _put(win, box_y + box_h - 1, box_x + 2, "└" + "─" * (box_w - 4) + "┘", attr_box)

    # Content
    # title set above based on active gate
    _put(win, box_y + 1, box_x + (box_w - len(title)) // 2,
         title, attr_box)
    _put(win, box_y + 2, box_x + 3,
         prompt[:box_w - 6], attr_inner)
    _put(win, box_y + 3, box_x + 3,
         f"Type  y + Enter  or  n + Enter",
         curses.color_pair(C_DIM))
    # Input echo
    input_display = f"> {state.modal_input}_"
    _put(win, box_y + 4, box_x + 3, input_display, attr_box)


def draw_flash(win, state: AppState, y: int, width: int):
    if state.last_action and (time.time() - state.last_action_time) < 1.2:
        msg = f"  ▶  {state.last_action}  "
        _put(win, y, max(0, (width - len(msg)) // 2), msg,
             curses.color_pair(C_ROBOT) | curses.A_BOLD)


def draw_log(win, state: AppState, y_start: int, height: int, width: int):
    _put(win, y_start, 2, "LOG", curses.color_pair(C_DIM) | curses.A_BOLD)
    avail = height - y_start - 2
    with state._lock:
        lines = state.log_lines[-avail:] if avail > 0 else []
    for idx, line in enumerate(lines):
        row = y_start + 1 + idx
        if row >= height - 1:
            break
        if "✓" in line:    pair = C_UNLOCKED
        elif "⚠" in line:  pair = C_WARN
        elif "↺" in line:  pair = C_RESET
        elif "modal" in line or "queued" in line: pair = C_QUEUE
        elif "running" in line: pair = C_RUNNING
        else:               pair = C_LOG
        _put(win, row, 2, line[: width - 4], curses.color_pair(pair))


def draw_nav_hints(win, state: AppState, y: int):
    if state.in_modal():
        col = 2
        # Slide nav still works – show it in normal colour
        col = draw_badge(win, y, col, "← H", "prev") + 3
        col = draw_badge(win, y, col, "→ L", "next") + 3
        # Then show what modal expects
        _put(win, y, col,
             "│  UNSTACK WAITING:  Y + Enter  /  N + Enter  to answer",
             curses.color_pair(C_MODAL) | curses.A_BOLD)
        return
    col = 2
    for key, label in [("← H", "prev"), ("→ L", "next"),
                        ("1", "unstack"), ("2", "grasp"),
                        ("3", "unfold"), ("4", "bagopen"),
                        ("R", "reset"), ("Q", "quit")]:
        col = draw_badge(win, y, col, key, label) + 3


def redraw(stdscr, state: AppState):
    stdscr.erase()
    h, w = stdscr.getmaxyx()

    ros_tag = " ROS ✓ " if state.ros_ready else " ROS … "
    header  = " Robot Operation Controller "
    _put(stdscr, 0, 0, " " * (w - 1), curses.color_pair(C_TITLE))
    _put(stdscr, 0, max(0, (w - len(header)) // 2), header,
         curses.color_pair(C_TITLE) | curses.A_BOLD)
    _put(stdscr, 0, w - len(ros_tag) - 1, ros_tag, curses.color_pair(C_TITLE))

    gated   = state.gated_slides()
    locked  = [s for s, st in gated.items() if st == 'locked']
    running = [s for s, st in gated.items() if st == 'running']
    queued  = state.queued_slide
    parts   = [f"  Slide: {state.current_slide}/{N_SLIDES}"]
    if locked:  parts.append(f"  locked:{sorted(locked)}")
    if running: parts.append(f"  running:{sorted(running)}")
    if queued:  parts.append(f"  queued→{queued}")
    _put(stdscr, 2, 0, "".join(parts), curses.color_pair(C_SLIDE))

    draw_slide_bar(stdscr, state, 3, w)
    _put(stdscr, 5, 0, "─" * min(w - 1, 78), curses.color_pair(C_DIM))
    draw_robot_panel(stdscr, state, 6, w)
    draw_flash(stdscr, state, 15, w)
    _put(stdscr, 16, 0, "─" * min(w - 1, 78), curses.color_pair(C_DIM))
    draw_log(stdscr, state, 17, h, w)
    draw_nav_hints(stdscr, state, h - 1)

    if state.in_modal():
        draw_modal(stdscr, state, h, w)

    stdscr.refresh()


# ═══════════════════════════════════════════════════════════════════════════
#  Main event loop
# ═══════════════════════════════════════════════════════════════════════════

def main_loop(stdscr, state: AppState, node: RobotNode,
              stack_node, bag_node, mh2):
    init_colors()
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)


    state.add_log("TUI started  –  1-4 modules  ← → slides  R reset  Q quit")

    # Redirect stdout so that print() calls from unstack (and anywhere else)
    # go to the TUI log instead of corrupting the curses display.
    class _LogWriter(io.TextIOBase):
        def write(self, s):
            s = s.strip()
            if s:
                state.add_log(f"  [stdout] {s}")
            return len(s)
        def flush(self): pass

    _real_stdout = sys.stdout
    _real_stderr = sys.stderr
    sys.stdout   = _LogWriter()
    sys.stderr   = _LogWriter()   # catches Python ROS logger output

    def flash(label: str):
        state.last_action      = label
        state.last_action_time = time.time()
        state.add_log(f"→ {label}")

    def show_slide(n: int):
        state.current_slide = n
        node.set_display(n)
        flash(f"Slide {n}")

    def go_slide(n: int):
        if not (1 <= n <= N_SLIDES):
            return
        # Gate only applies when moving FORWARD into a locked slide.
        # Moving backward (n < current) is always allowed so you can
        # freely navigate back through already-seen slides.
        going_forward = n > state.current_slide
        status = state.gated_slides().get(n, 'open')
        if status == 'open' or not going_forward:
            show_slide(n)
        elif status == 'running':
            with state._lock:
                state.queued_slide = n
            state.add_log(f"  slide {n} queued – will show when module finishes")
        else:
            nonlocal _last_warn_key, _last_warn_time
            warn_key = f"locked-{n}"
            now = time.time()
            if warn_key != _last_warn_key or now - _last_warn_time > 2.0:
                state.add_log(f"⚠  slide {n} locked – trigger the module first (3/4)")
                _last_warn_key  = warn_key
                _last_warn_time = now

    def on_slide_ready(n: int):
        show_slide(n)

    node._slide_ready = on_slide_ready

    def start_unstack():
        if state.unstack_running:
            state.add_log("⚠  UNSTACK already running")
            return
        with state._lock:
            state.unstack_running = True
            state.modal_input     = ""
        flash("UNSTACK")
        threading.Thread(
            target=tui_unstack_utils.run,
            args=(state, stack_node, mh2),
            daemon=True
        ).start()

    _last_warn_key  = None   # debounce: suppress repeated warning for same key
    _last_warn_time = 0.0

    while True:
        redraw(stdscr, state)
        time.sleep(0.05)

        ch = stdscr.getch()
        if ch == -1:
            _last_warn_key = None   # reset debounce when key released
            continue

        # ── Modal input: Y/N/Backspace/Enter handled first ───────────────
        # Slide navigation (arrows / H / L) still works during modal.
        # Module triggers (1-4), reset (R), and quit (Q) are blocked.
        if state.in_modal():
            if ch in (curses.KEY_BACKSPACE, 127, 8):
                state.modal_input = state.modal_input[:-1]
                continue
            elif ch in (ord('\n'), ord('\r'), curses.KEY_ENTER):
                answer = state.modal_input.strip().lower()
                state.add_log(f"  [modal] answered: {answer!r}")
                state.modal_input = ""
                gate = state.active_gate()
                if gate:
                    gate.answer(answer)
                continue
            elif 32 <= ch < 127:
                c = chr(ch).lower()
                if c in ('y', 'n'):
                    state.modal_input += chr(ch)
                    continue
            # fall through to slide navigation below; all other keys ignored

        # ── Slide navigation – always active ─────────────────────────────
        if ch in (curses.KEY_RIGHT, ord('l'), ord('L')):
            go_slide(state.current_slide + 1)
            continue

        elif ch in (curses.KEY_LEFT, ord('h'), ord('H')):
            go_slide(state.current_slide - 1)
            continue

        # ── Everything below is blocked while modal is open ───────────────
        if state.in_modal():
            continue

        if ch in (ord('q'), ord('Q')):
            sys.stdout = _real_stdout
            sys.stderr = _real_stderr
            break

        elif ch == ord('1'):
            start_unstack()

        elif ch == ord('2'):
            flash("GRASP")
            node.trigger_grasp()

        elif ch == ord('3'):
            flash("UNFOLD")
            node.trigger_unfold()

        elif ch == ord('4'):
            if state.bagopen_running:
                state.add_log("⚠  BAG OPEN already running")
            else:
                with state._lock:
                    state.bagopen_running = True
                    state.modal_input     = ""
                flash("BAG OPEN")
                threading.Thread(
                    target=tui_bagopen_utils.run,
                    args=(state, node, bag_node, mh2),
                    daemon=True
                ).start()

        elif ch in (ord('r'), ord('R')):
            state.do_reset()
            node.publish_reset()
            flash("RESET")
            show_slide(1)   # update display + TTS for slide 1


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import os


    state = AppState()

    start_slide = 1
    if len(sys.argv) > 1:
        try:
            v = int(sys.argv[-1])
            assert 1 <= v <= N_SLIDES
            start_slide = v
        except (ValueError, AssertionError):
            pass
    state.current_slide = start_slide

    rclpy.init()

    node = RobotNode(state, on_slide_ready=lambda n: None)

    # StackDetectorDINO and TrajectoryPublisher – always with_slides=True
    # MotionHelperV2 is shared between both.
    mh2        = MotionHelperV2()
    stack_node = tui_unstack_utils.StackDetectorDINO(with_slides=True)
    bag_node   = tui_bagopen_utils.TrajectoryPublisher(with_slides=False)

    # Spin all nodes on a shared multi-threaded executor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(stack_node)
    executor.add_node(bag_node)
    threading.Thread(target=executor.spin, daemon=True).start()

    state.add_log("StackDetectorDINO ready")

    try:
        curses.wrapper(main_loop, state, node, stack_node, bag_node, mh2)
    finally:
        node.destroy_node()
        stack_node.destroy_node()
        bag_node.destroy_node()
        rclpy.shutdown()

    print("TUI closed.")


if __name__ == "__main__":
    main()