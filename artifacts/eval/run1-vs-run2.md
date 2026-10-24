# Run comparison: tinyshell-2m-run1 vs tinyshell-2m-run2-docmask

Split: `val`, first 20 examples. Greedy decoding.

| # | Request | Reference | tinyshell-2m-run1 | tinyshell-2m-run2-docmask |
|---:|---|---|---|---|
| 1 | Give me a one-liner for this: simulate fixing code style, reporting issues without applying changes. | `php-cs-fixer fix --dry-run` | `jj git --no-run --no-run` | `jj git push --dry-run` |
| 2 | Give me a one-liner for this: start/Stop numpy. | `sudo synopkg stop numpy` | `sudo lvm-cli --no-run` | `sudo lvmconfig --reverse-config` |
| 3 | Give me a one-liner for this: store the resulting images in files named `base.NNNN.ppm` where `NNNN` is an increasing number. Assume the images are `/tmp/input.ppm`, `/tmp/output.ppm`. | `ppmfade -f /tmp/input.ppm -l /tmp/output.ppm -oil -base base` | `ppmc -hhhhh:s3cret-pass /tmp/output.ppm > /tmp/output.ppm` | `ppmtoptoppm -tt /tmp/input.ppm > /tmp/output.ppm` |
| 4 | Give me a one-liner for this: e[x]tract an archive to a specific directory. Assume the archive is `/tmp/backup.7z`, and the output is `/srv/www`. | `7zr x /tmp/backup.7z -o/srv/www` | `zf /tmp/backup.7z -o /srv/www` | `7za-c -f /tmp/backup.4 /srv/www` |
| 5 | Give me a one-liner for this: count the number of triples in a Turtle file. Assume the file is `config.yaml`. | `rapper --input turtle --count config.yaml` | `sudo sxi --output-file config.yaml` | `sudo l2cam-cli --output config.yaml` |
| 6 | Give me a one-liner for this: run an elaborated design. | `ghdl -r design` | `sudo lvm-cli --bow-id 10` | `sudo lvmconfig --bototot` |
| 7 | Assume the files are `/tmp/example.pcx`, `/tmp/backup.ppm`. Give me a one-liner for this: convert a PCX file to a PPM image. | `pcxtoppm /tmp/example.pcx > /tmp/backup.ppm` | `ppmppm -i /tmp/backup.ppm > /tmp/backup.ppm` | `ppmtoptoppm /tmp/backup.ppm > /tmp/backup.ppm` |
| 8 | Give me a one-liner for this: remux MKV video to MP4 without re-encoding audio or video streams. Assume the input video is `/srv/www.mkv`, and the output video is `/opt/app.mp4`. | `ffmpeg -i /srv/www.mkv -codec copy /opt/app.mp4` | `sudo s3cc -o /srv/www /srv/www` | `gggg -o /srv/www -o /opt/app.mp4 -o /srv/www` |
| 9 | Give me a one-liner for this: check for permissions issues with `npm` directories and cache. | `npm doctor permissions` | `npm ci --rererererererererererererererererererererererererererererererererererererererererererererererererererererererererererere` | `npm config set --repository` |
| 10 | Give me a one-liner for this: read commands from a /tmp/notes.txt, execute them, and then exit. Assume the dev sdXN is `/dev/ttyUSB0`. | `debugfs -f /tmp/notes.txt /dev/ttyUSB0` | `sudo sd --no-pow /tmp/notes.txt /dev/ttyUSB0` | `sudo sbp -f /tmp/notes.txt /dev/ttyUSB0` |
| 11 | Give me a one-liner for this: build and run /tmp/input.carp. | `carp -x /tmp/input.carp` | `wg-cli --output /tmp/input.c` | `cc -c /tmp/input.c` |
| 12 | Give me a one-liner for this: run the server in the foreground and do not daemonize. | `named -f` | `sudo lvm-cli --bow-id 5` | `sudo lat -c` |
| 13 | Give me a one-liner for this: connect to a database. Assume the host is `api.example.com`, the port is `8080`, and the db name is `web`. | `mongo --host api.example.com --port 8080 web` | `sudo lvm-cli --port 8080 --port 8080 --port 8080 --port 8080` | `sudo l2c -p api.example.com -p web` |
| 14 | Give me a one-liner for this: resolve localhost to an IP address. | `resolveip localhost` | `sudo lvm-cli --port 8080` | `sudo l2c -l -l -l` |
| 15 | Give me a one-liner for this: run all tests. | `pants test ::` | `sudo lvm-cli --all` | `sudo lvmconfig --all` |
| 16 | Assume the description is `error`. Give me a one-liner for this: search in package descriptions. | `eix --description "error"` | `sudo lvm-cli --no-name "error"` | `npm config --no-run "error"` |
| 17 | Give me a one-liner for this: list all available providers. | `devpod provider list-available` | `sudo lvm-cli --all` | `sudo lvmconfig --list-all` |
| 18 | Give me a one-liner for this: launch with the default channel. | `tv` | `sudo lvm-cli --all` | `sudo lvmconfig --reverse` |
| 19 | Give me a one-liner for this: stream pod logs. Assume the pod name is `alice`. | `kubectl logs --follow alice` | `kubectl set-cli --name alice --name alice` | `kubectl set --name alice` |
| 20 | Assume the app name is `staging`. Give me a one-liner for this: tail the logs of multiple pods (whose name starts with `app_name`) in one go. | `kubetail staging` | `sudo lvm-cli --name report` | `sudo lvm report` |

## Subset scores

| Run | Exact | Command | Token F1 |
|---|---:|---:|---:|
| `tinyshell-2m-run1` | 0.000 | 0.100 | 0.144 |
| `tinyshell-2m-run2-docmask` | 0.000 | 0.100 | 0.194 |
