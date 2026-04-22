#!/bin/bash
# ==========================================
# Git 自动同步 + 保护脚本
# 功能：
#   1. 检查仓库完整性
#   2. 检测变更 → 自动提交 → 推送到 GitHub
#   3. 自动备份
# ==========================================

REPO_DIR="/Users/chixinning/Desktop/catpaw工作台/sft_demo_for_interview"
BACKUP_DIR="/Users/chixinning/.repo_backup/sft_demo_for_interview"
LOG_FILE="$REPO_DIR/.auto_sync.log"
COMMIT_MSG="Auto sync: $(date '+%Y-%m-%d %H:%M:%S')"

# 记录日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# ==========================================
# 1. 检查仓库完整性
# ==========================================
check_integrity() {
    if [ ! -d "$REPO_DIR/.git" ]; then
        log "⚠️ 警告: .git 目录丢失！"
        
        # 从备份恢复
        if [ -d "$BACKUP_DIR/.git" ]; then
            log "正在从备份恢复 .git..."
            cp -R "$BACKUP_DIR/.git" "$REPO_DIR/"
            log "✅ .git 已恢复"
        else
            log "❌ 无可用备份，请手动克隆仓库"
            exit 1
        fi
    fi
}

# ==========================================
# 2. 同步到 GitHub
# ==========================================
check_and_sync() {
    cd "$REPO_DIR" || exit 1
    
    # 检查是否有未跟踪或已修改的文件
    CHANGES=$(git status --porcelain 2>/dev/null)
    
    if [ -n "$CHANGES" ]; then
        log "检测到变更，开始同步..."
        
        # 添加所有变更
        git add -A
        
        # 提交
        git commit -m "$COMMIT_MSG"
        
        # 推送
        git push 2>&1
        
        if [ $? -eq 0 ]; then
            log "✅ 同步成功: $COMMIT_MSG"
        else
            log "❌ 推送失败"
        fi
    fi
}

# ==========================================
# 3. 创建备份
# ==========================================
create_backup() {
    mkdir -p "$BACKUP_DIR"
    rsync -a --delete "$REPO_DIR/" "$BACKUP_DIR/" 2>/dev/null
    log "📦 备份完成"
}

# ==========================================
# 主流程
# ==========================================
check_integrity
check_and_sync
create_backup
