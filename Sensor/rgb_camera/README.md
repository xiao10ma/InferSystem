# RGB Camera

This folder owns rgb camera abstractions and drivers.

`BaseRGBCamera` currently provides the common lifecycle and parameter API:

- `open()`
- `close()`
- `restart()`
- `register_stream()`
- `configure_stream()`
- `set_param()`
- `set_params()`
- `get_param()`
- `get_params()`
- `build_frame()`
- `read_frame()`

Concrete camera drivers should inherit from this base class and only override
hardware-specific behavior.

Design notes:

- Device control params and video stream configs are separated.
- `color` and `depth` can coexist in one camera object.
- New camera vendors can map their SDK options into shared param keys such as
  `exposure.auto` or `white_balance.auto`.
