# Example 21 retirement

## Scope

Example 21 has one supported executable path: `examples/pp_prop/21-braincell-arc.py`.
The replacement implementation owns its raw ARC inputs and must keep those files
available to the image without building an index or source manifest.

The retired path includes the latent-workspace modules and tests, the old ARC
entry points, diagnostic scripts, the ARC index builder and tests, generated
index commands, and Docker metadata that describes indexed or manifest-backed
data.

## Compatibility

The public `braintrace` package API is unchanged. This retirement changes only
Example 21 support files, documentation, and container wiring.

## Verification

The retirement is complete when repository scans find no import or command for a
retired module, the Example 21 README and image command name the BrainCell
entry point, Docker preserves raw ARC files, and importing the public package
still succeeds.
