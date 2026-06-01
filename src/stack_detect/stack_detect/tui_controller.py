"""
tui_controller.py  –  Robot operation TUI
==========================================
Replaces slideshow.py with a full-terminal curses interface.

Key bindings
------------
  Slides      →  / L   : next slide          ←  / H   : previous slide
  Modules     F1        : UNSTACK             F2        : GRASP
              F3        : UNFOLD              F4        : BAG OPEN
  Reset       F10       : reset unfold/bagopen flags + queue  (far-right F-key)
  Quit        Q

Slide-gate logic
----------------
  UNFOLD_DONE_SLIDE  is gated on /unfold_done   publishing True.
  BAGOPEN_DONE_SLIDE is gated on /bag_open_done publishing True.

  Flag is STICKY once True – navigating to the gated slide is instant.
  If the module is currently running (triggered but not yet done) and you
  press → toward the gated slide, the intent is QUEUED and fires automatically
  when the done signal arrives.
  If the module has never been triggered, pressing → toward the gated slide
  just warns – no queue.

  F10 resets both flags to False, clears running state and any queued nav,
  and publishes False on /unfold_done and /bag_open_done so the modules know
  the state has been cleared.
"""

import sys
import curses
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

# ── ROS2 imports (gracefully degrade for testing without ROS) ─────────────
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from softenable_display_msgs.srv import SetDisplay



# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  –  edit these to match your setup
# ═══════════════════════════════════════════════════════════════════════════

N_SLIDES            = 12

# Which slide is gated on each module finishing?  None = no gate.
UNFOLD_DONE_SLIDE   = 7
BAGOPEN_DONE_SLIDE  = 10

# ROS topic / service names
TOPIC_UNFOLD_DONE   = "/unfold_done"
TOPIC_BAGOPEN_DONE  = "/bag_open_done"
SRV_SET_DISPLAY     = "/set_display"
SRV_UNSTACK         = "/unstack/trigger"
SRV_GRASP           = "/grasp/trigger"
SRV_UNFOLD          = "/unfold/trigger"
SRV_BAGOPEN         = "/bag_open/trigger"

SLIDE_PREFIX        = "study"   # slides named study-1 … study-N


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
C_QUEUE    = 10   # cyan – queued-slide indicator
C_RUNNING  = 11   # magenta – module currently executing
C_RESET    = 12   # red-on-white – reset key badge


# ═══════════════════════════════════════════════════════════════════════════
#  Application state  (shared between ROS thread and curses thread)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AppState:
    current_slide:    int            = 1

    # Sticky completion flags – set True by topic, reset only by F10
    unfold_done:      bool           = False
    bagopen_done:     bool           = False

    # Running flags – True from trigger until done callback fires
    unfold_running:   bool           = False
    bagopen_running:  bool           = False

    # A single queued slide intent (set when navigating toward a locked slide
    # while its module is running; cleared on delivery or reset)
    queued_slide:     Optional[int]  = None

    log_lines:        list           = field(default_factory=list)
    ros_ready:        bool           = False
    last_action:      str            = ""
    last_action_time: float          = 0.0

    # Lock for state fields touched by both threads
    _lock:            object         = field(default_factory=threading.Lock,
                                            repr=False, compare=False)

    # ── helpers ──────────────────────────────────────────────────────────

    def gated_slides(self) -> dict:
        """Return {slide_no: status} for each gated slide.
        status: 'open' | 'running' | 'locked'
        """
        result = {}
        if UNFOLD_DONE_SLIDE:
            if self.unfold_done:
                result[UNFOLD_DONE_SLIDE] = 'open'
            elif self.unfold_running:
                result[UNFOLD_DONE_SLIDE] = 'running'
            else:
                result[UNFOLD_DONE_SLIDE] = 'locked'
        if BAGOPEN_DONE_SLIDE:
            if self.bagopen_done:
                result[BAGOPEN_DONE_SLIDE] = 'open'
            elif self.bagopen_running:
                result[BAGOPEN_DONE_SLIDE] = 'running'
            else:
                result[BAGOPEN_DONE_SLIDE] = 'locked'
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
        self.add_log("↺ RESET – flags cleared, slides re-locked, queue cleared")


# ═══════════════════════════════════════════════════════════════════════════
#  ROS2 node  (spun in a daemon thread)
# ═══════════════════════════════════════════════════════════════════════════

class RobotNode(Node):
    def __init__(self, state: AppState, on_slide_ready):
        """
        on_slide_ready(n) is called from the ROS thread when a queued slide
        should be displayed (module just finished).
        """
        super().__init__("tui_controller")
        self.state         = state
        self._slide_ready  = on_slide_ready

        self.cli_display = self.create_client(SetDisplay, SRV_SET_DISPLAY)
        self.cli_unstack = self.create_client(Trigger,    SRV_UNSTACK)
        self.cli_grasp   = self.create_client(Trigger,    SRV_GRASP)
        self.cli_unfold  = self.create_client(Trigger,    SRV_UNFOLD)
        self.cli_bagopen = self.create_client(Trigger,    SRV_BAGOPEN)

        if UNFOLD_DONE_SLIDE:
            self.create_subscription(Bool, TOPIC_UNFOLD_DONE,
                                     self._cb_unfold_done, 10)
        if BAGOPEN_DONE_SLIDE:
            self.create_subscription(Bool, TOPIC_BAGOPEN_DONE,
                                     self._cb_bagopen_done, 10)

        state.ros_ready = True
        state.add_log("ROS2 node ready")

    # ── topic callbacks ───────────────────────────────────────────────────

    def _cb_unfold_done(self, msg: Bool):
        if not msg.data:
            return
        with self.state._lock:
            if self.state.unfold_done:
                return                       # already flagged – ignore repeat
            self.state.unfold_done    = True
            self.state.unfold_running = False
            queued = self.state.queued_slide
            if queued == UNFOLD_DONE_SLIDE:
                self.state.queued_slide = None
        self.state.add_log(
            f"✓ Unfold finished  →  slide {UNFOLD_DONE_SLIDE} unlocked")
        if queued == UNFOLD_DONE_SLIDE:
            self.state.add_log(
                f"  queued intent fired  →  showing slide {UNFOLD_DONE_SLIDE}")
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
        self.state.add_log(
            f"✓ Bag-open finished  →  slide {BAGOPEN_DONE_SLIDE} unlocked")
        if queued == BAGOPEN_DONE_SLIDE:
            self.state.add_log(
                f"  queued intent fired  →  showing slide {BAGOPEN_DONE_SLIDE}")
            self._slide_ready(BAGOPEN_DONE_SLIDE)

    # ── service calls ─────────────────────────────────────────────────────

    def set_display(self, slide_no: int):
        if not self.cli_display.service_is_ready():
            self.state.add_log("⚠  /set_display not available")
            return
        req = SetDisplay.Request()
        req.name    = f"{SLIDE_PREFIX}-{slide_no}"
        req.use_tts = True
        future = self.cli_display.call_async(req)
        future.add_done_callback(
            lambda f: self.state.add_log(
                f"  slide {slide_no} display → "
                f"{'ok' if f.result() and f.result().success else 'FAIL'}"
            )
        )

    def _trigger(self, client, label: str, done_cb=None):
        if not client.service_is_ready():
            self.state.add_log(f"⚠  {label} service not available")
            return
        future = client.call_async(Trigger.Request())
        def _cb(f):
            ok = f.result() and f.result().success
            self.state.add_log(
                f"  {label} trigger → {'ok' if ok else 'FAIL'}"
            )
            if done_cb:
                done_cb(ok)
        future.add_done_callback(_cb)

    def trigger_unstack(self):
        self._trigger(self.cli_unstack, "UNSTACK")

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
        tag = f"{i:2d}"

        if is_current:
            attr = curses.color_pair(C_KEY) | curses.A_BOLD
            marker = f"[{tag}]"
        elif is_queued:
            attr = curses.color_pair(C_QUEUE) | curses.A_BOLD
            marker = f">{tag}<"
        elif status == 'running':
            attr = curses.color_pair(C_RUNNING)
            marker = f" {tag}~"
        elif status == 'locked':
            attr = curses.color_pair(C_LOCKED)
            marker = f" {tag}L"
        else:
            attr = curses.color_pair(C_SLIDE)
            marker = f" {tag} "

        _put(win, y, col, marker, attr)
        col += len(marker) + 1
        if col > width - 5:
            break


def draw_robot_panel(win, state: AppState, y: int, width: int):
    _put(win, y, 2, "ROBOT MODULES", curses.color_pair(C_ROBOT) | curses.A_BOLD)

    # Row 1  –  UNSTACK  and  GRASP  (no gate, always available)
    r1 = y + 2
    draw_badge(win, r1, 4,  "F1", "UNSTACK")
    draw_badge(win, r1, 28, "F2", "GRASP")

    # Row 2  –  UNFOLD  and  BAG OPEN  (gated)
    r2 = y + 4
    draw_badge(win, r2, 4,  "F3", "UNFOLD")
    _draw_gate_hint(win, r2, 20, UNFOLD_DONE_SLIDE,
                    state.unfold_done, state.unfold_running)

    draw_badge(win, r2, 28, "F4", "BAG OPEN")
    _draw_gate_hint(win, r2, 46, BAGOPEN_DONE_SLIDE,
                    state.bagopen_done, state.bagopen_running)

    # Row 3  –  RESET  (visually separated, red badge to stand out)
    r3 = y + 6
    _put(win, r3, 2,
         "─" * min(width - 3, 70),
         curses.color_pair(C_DIM))
    draw_badge(win, r3 + 1, 4, "F10", "RESET  (clears flags, re-locks slides, drops queue)",
               pair=C_RESET)


def _draw_gate_hint(win, y, x, slide_no, done: bool, running: bool):
    if not slide_no:
        return
    if done:
        _put(win, y, x,
             f"slide {slide_no}: unlocked",
             curses.color_pair(C_UNLOCKED))
    elif running:
        _put(win, y, x,
             f"slide {slide_no}: running…",
             curses.color_pair(C_RUNNING) | curses.A_BOLD)
    else:
        _put(win, y, x,
             f"slide {slide_no}: waiting",
             curses.color_pair(C_LOCKED))


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
        if "✓" in line:
            pair = C_UNLOCKED
        elif "⚠" in line:
            pair = C_WARN
        elif "↺" in line:
            pair = C_RESET
        elif "queued" in line or ">" in line:
            pair = C_QUEUE
        elif "running" in line or "~" in line:
            pair = C_RUNNING
        else:
            pair = C_LOG
        _put(win, row, 2, line[: width - 4], curses.color_pair(pair))


def draw_nav_hints(win, y: int):
    col = 2
    for key, label in [("← H", "prev"), ("→ L", "next"),
                        ("F1-F4", "modules"), ("F10", "reset"), ("Q", "quit")]:
        col = draw_badge(win, y, col, key, label) + 4


def redraw(stdscr, state: AppState):
    stdscr.erase()
    h, w = stdscr.getmaxyx()

    # Header
    ros_tag = " ROS ✓ " if state.ros_ready else " ROS … "
    header  = " Robot Operation Controller "
    _put(stdscr, 0, 0, " " * (w - 1),   curses.color_pair(C_TITLE))
    _put(stdscr, 0, max(0, (w - len(header)) // 2), header,
         curses.color_pair(C_TITLE) | curses.A_BOLD)
    _put(stdscr, 0, w - len(ros_tag) - 1, ros_tag, curses.color_pair(C_TITLE))

    # Slide status + bar
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

    draw_nav_hints(stdscr, h - 1)

    stdscr.refresh()


# ═══════════════════════════════════════════════════════════════════════════
#  Main event loop
# ═══════════════════════════════════════════════════════════════════════════

def main_loop(stdscr, state: AppState, node: Optional["RobotNode"]):
    init_colors()
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    state.add_log("TUI started  –  F1-F4 modules  ← → slides  F10 reset  Q quit")

    def flash(label: str):
        state.last_action      = label
        state.last_action_time = time.time()
        state.add_log(f"→ {label}")

    def show_slide(n: int):
        """Unconditionally display slide n (gate already satisfied)."""
        state.current_slide = n
        if node:
            node.set_display(n)
        flash(f"Slide {n}")

    def go_slide(n: int):
        """Navigate to slide n, respecting gate logic."""
        if not (1 <= n <= N_SLIDES):
            return
        gated = state.gated_slides()
        status = gated.get(n, 'open')

        if status == 'open':
            # Flag already True (sticky) – go immediately
            show_slide(n)

        elif status == 'running':
            # Module triggered but not done – store intent, fire when done
            with state._lock:
                state.queued_slide = n
            state.add_log(
                f"  slide {n} queued – will show when module finishes")

        else:
            # Module never triggered – block and warn
            state.add_log(
                f"⚠  slide {n} locked – trigger the module first (F3/F4)")

    # Callback passed to RobotNode so it can call show_slide from ROS thread
    # (writing state.current_slide is safe; curses thread only reads it)
    def on_slide_ready(n: int):
        show_slide(n)

    if node:
        node._slide_ready = on_slide_ready

    while True:
        redraw(stdscr, state)
        time.sleep(0.05)

        ch = stdscr.getch()
        if ch == -1:
            continue

        if ch in (ord('q'), ord('Q')):
            break

        elif ch in (curses.KEY_RIGHT, ord('l'), ord('L')):
            go_slide(state.current_slide + 1)

        elif ch in (curses.KEY_LEFT, ord('h'), ord('H')):
            go_slide(state.current_slide - 1)

        elif ch == curses.KEY_F1:
            flash("UNSTACK")
            if node: node.trigger_unstack()

        elif ch == curses.KEY_F2:
            flash("GRASP")
            if node: node.trigger_grasp()

        elif ch == curses.KEY_F3:
            flash("UNFOLD")
            if node: node.trigger_unfold()

        elif ch == curses.KEY_F4:
            flash("BAG OPEN")
            if node: node.trigger_bagopen()

        elif ch == curses.KEY_F10:
            state.do_reset()
            flash("RESET")


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
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

    node = None
    rclpy.init()
    # on_slide_ready is patched in main_loop once curses is running
    node = RobotNode(state, on_slide_ready=lambda n: None)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    curses.wrapper(main_loop, state, node)
    if node:
        node.destroy_node()
        rclpy.shutdown()

    print("TUI closed.")


if __name__ == "__main__":
    main()