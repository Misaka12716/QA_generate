#!/usr/bin/env bash
# 首次与 GitHub 同步（推荐 SSH：见 scripts/github-ssh-key.pub 添加到 GitHub）
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> fetch origin/main"
git fetch origin

if ! git rev-parse --verify origin/main >/dev/null 2>&1; then
  echo "未找到 origin/main，请检查远程仓库与网络。" >&2
  exit 1
fi

if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  echo "==> 本地尚无提交，对齐远程 main（保留工作区未跟踪/本地修改）"
  git reset --mixed origin/main
  echo "完成。可用 git status 查看，再 git add / commit / push。"
  exit 0
fi

echo "==> 拉取并变基到 origin/main"
git pull --rebase origin main
echo "完成。"
