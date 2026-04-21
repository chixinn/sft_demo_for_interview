#!/bin/bash
# ==========================================
# Git 自动同步脚本
# 功能：检测变更 → 自动提交 → 推送到 GitHub
# ==========================================

REPO_DIR="/Users/chixinning/Desktop/catpaw工作台/sft_demo_for_interview"
LOG_FILE="$REPO_DIR/.auto_sync.log"
COMMIT_MSG="Auto sync: $(date '+%Y-%m-%d %H:%M:%S')"

cd "$REPO_DIR" || exit 1

# 记录日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# 检查是否有变更
check_and_sync() {
    # 检查是否有未跟踪或已修改的文件
    CHANGES=$(git status --porcelain 2>/dev/null)
    
    if [ -n "$CHANGES" ]; then
        log "检测到变更，开始同步..."
        
        # 添加所有变更
        git add -A
        
        # 提交
        git commit -m "$COMMIT_MSG"
        
        # 推送
        PUSH_RESULT=$(git push 2>&1)
        
        if [ $? -eq 0 ]; then
            log "✅ 同步成功: $COMMIT_MSG"
        else
            log "❌ 推送失败: $PUSH_RESULT"
        fi
    fi
}

# 执行同步
check_and_sync
