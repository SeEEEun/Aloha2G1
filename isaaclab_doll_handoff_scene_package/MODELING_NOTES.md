# Modeling notes

- Existing authoritative table values are taken from the project's scene layout: 0.835 m x 0.720 m, tabletop z=0.795 m, black edge rail width 0.025 m.
- The supplied Daiso pages returned HTTP 403 to automated retrieval, so exact listed product dimensions could not be verified from the URLs.
- Doll/ball and bin dimensions are therefore photo-scaled provisional values, isolated in one JSON file for easy replacement after direct measurement.
- The bin is modeled as an open container with four collision walls and a bottom; there is intentionally no collider across the opening.
- The G1 0.15 m request is interpreted as pelvis/root forward distance from the table front edge. Only the copied doll-handoff scene is patched.
