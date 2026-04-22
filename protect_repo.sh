#!/bin/bash
# ==========================================
# Git 仓库保护脚本
# 功能：防止 .git 目录和重要文件被误删
# ==========================================

REPO_DIR="/Users/chixinning/Desktop/catpaw工作台/sft_demo_for_interview"
BACKUP_DIR="/Users/chixinning/.repo_backup/sft_demo_for_interview"
LOG_FILE="$REPO_DIR/.protection.log"

# 记录日志
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# 1. 设置文件保护属性（防止误删）
set_protection() {
    log "设置文件保护..."
    
    # 保护 .git 目录
    chflags -R uchg "$REPO_DIR/.git" 2>/dev/null && log "✅ .git 目录已加锁"
    
    # 保护关键 Python 文件
    for file in "$REPO_DIR"/*.py; do
        if [ -f "$file" ]; then
            chflags uchg "$file" 2>/dev/null
        fi
    done
    log "✅ Python 文件已加锁"
    
    # 保护关键数据文件
    for ext in json jsonl; do
        for file in "$REPO_DIR"/*.$ext; do
            if [ -f "$file" ]; then
                chflags uchg "$file" 2>/dev/null
            fi
        done
    done
    log "✅ 数据文件已加锁"
}

# 2. 创建备份
create_backup() {
    log "创建备份..."
    
    mkdir -p "$BACKUP_DIR"
    
    # 备份整个仓库（包括 .git）
    rsync -a --delete "$REPO_DIR/" "$BACKUP_DIR/" 2>/dev/null
    
    log "✅ 备份完成: $BACKUP_DIR"
}

# 3. 检查仓库完整性
check_integrity() {
    if [ ! -d "$REPO_DIR/.git" ]; then
        log "⚠️ 警告: .git 目录丢失！"
        
        # 从备份恢复
        if [ -d "$BACKUP_DIR/.git" ]; then
            log "正在从备份恢复..."
            cp -R "$BACKUP_DIR/.git" "$REPO_DIR/"
            log "✅ .git 已恢复"
        fi
    fi
}

# 主流程
case "$1" in
    "lock")
        set_protection
        ;;
    "backup")
        create_backup
        ;;
    "check")
        check_integrity
        ;;
    "all"|*)
        set_protection
        create_backup
        check_integrity
        ;;
esac
