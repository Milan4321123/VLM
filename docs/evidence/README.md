# Execution evidence

These files establish that the software executed. They are not evidence of
object-detection accuracy. No fine-tuning was performed.

- `cpu-preflight.json`: final CPU container import/device check, without weights.
- `cpu-acceptance.json` and `.jpg`: full pretrained Linux arm64 Docker pipeline,
  **one-token limit**. The response was `Based`; no region selection was parsed.
- `docker-tests.txt`: **47 passed** inside the tested CPU image.
- `docker-stages.txt`: image, region and generation timings from that run.
- `native-acceptance.json` and `.jpg`: full pretrained macOS CPU pipeline,
  **16-token limit**, with the Docker VM stopped. The response is truncated;
  no region selection was parsed.
- `native-stages.txt` and `native-environment.json`: native timing and environment.

The images have no detection rectangles because neither response contained a
valid region selection. JSON paths under `outputs/` refer to the original runs;
these are unchanged copies of those results. See [validation](../validation.md)
for the commands, checkpoint revisions, tested image identifier and limits.

After the 16 September cleanup, `cleanup-tests.txt` records 47 passing local
tests and `cleanup-preflight.json` records the no-download import/device check.
The Docker and full-model results above retain their original run dates.

On 17 September, `review-tests.txt` records 56 passing tests after tiny-image
box-scaling and prose-tag parsing fixes. `review-preflight.json` records the
CPU import/device check. These checks did not repeat full pretrained inference
or a Docker build.

A subsequent 17 September laptop test used the current source with a **32-token
limit**: `laptop-32-token.json`, `laptop-32-token.jpg` and
`laptop-32-token-stages.txt`. The application saved valid outputs in 231.785
seconds, but described a television in the fruit image and selected no regions.
This is a recorded qualitative failure, not successful detection. See the
follow-up section in the validation record for system resources and timing-tool
instrumentation limitations.
