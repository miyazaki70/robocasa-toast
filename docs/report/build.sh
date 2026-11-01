#!/bin/bash
# report.md から report.html と report.pdf を作る。
#   bash docs/report/build.sh
# 文章は report.md のとおり（pandoc で変換するだけで、文言は変えない）。見た目は template.html の CSS で決まる。
# 必要なもの: pandoc、Node.js と Playwright（build_pdf.js）
set -euo pipefail
# 日本語のタイトルを pandoc の引数で渡すので UTF-8 のロケールにする（POSIX のままだと <title> が文字化けする）
export LC_ALL=C.UTF-8
cd "$(dirname "$0")"
title=$(sed -n 's/^# //p' report.md | head -n 1)
pandoc report.md -f gfm -t html5 --wrap=preserve \
  --template template.html --metadata pagetitle="$title" -o report.html
node build_pdf.js report.html report.pdf
