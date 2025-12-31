# Git 误提交 Checkpoint 文件的解决方案

## 情况判断

首先需要判断你的情况：

1. **文件已暂存但未提交**（已解决）
2. **已提交但未推送到远程**
3. **已推送到远程**

---

## 方案一：已暂存但未提交（最简单）

如果文件只是 `git add` 了，还没有 `git commit`：

```bash
# 从暂存区移除文件（保留本地文件）
git restore --staged checkpoint/your_file.pth

# 或者使用旧版本命令
git reset HEAD checkpoint/your_file.pth
```

---

## 方案二：已提交但未推送到远程

### 2.1 如果是最近的提交（推荐）

如果 checkpoint 文件在**最新的 commit** 中：

```bash
# 方法1: 修改最近一次提交（推荐）
git reset --soft HEAD~1          # 撤销提交，保留更改在暂存区
git restore --staged checkpoint/ # 从暂存区移除checkpoint文件
git add .                        # 重新添加其他需要的文件
git commit -m "Your commit message"

# 方法2: 使用 git reset --mixed（默认）
git reset HEAD~1                 # 撤销提交，更改回到工作区
git restore --staged checkpoint/ # 从暂存区移除
# 然后重新添加和提交需要的文件
```

### 2.2 如果是倒数第N次提交

```bash
# 使用交互式rebase
git rebase -i HEAD~N  # N是提交的数量

# 在编辑器中，找到包含checkpoint的提交，将pick改为edit
# 保存退出后执行：
git restore --staged checkpoint/
git commit --amend --no-edit
git rebase --continue
```

---

## 方案三：已推送到远程（需要谨慎）

### 3.1 从版本控制中移除文件（保留本地文件）

```bash
# 从Git索引中移除文件，但保留本地文件
git rm --cached -r checkpoint/
git rm --cached -r checkpoints/

# 确保.gitignore包含这些规则
echo "checkpoint/" >> .gitignore
echo "checkpoints/" >> .gitignore
echo "*.pth" >> .gitignore  # 如果只想忽略pth文件
echo "*.pt" >> .gitignore
echo "*.ckpt" >> .gitignore

# 提交更改
git add .gitignore
git commit -m "Remove checkpoint files from version control"
git push
```

**⚠️ 警告**：这个方法会保留文件在Git历史中，只是之后不再跟踪。

### 3.2 完全从历史中移除（重写历史）

如果需要完全从Git历史中删除大文件：

```bash
# 使用 git filter-branch（较老的方法，不推荐）
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch -r checkpoint/ checkpoints/" \
  --prune-empty --tag-name-filter cat -- --all

# 推荐使用 git-filter-repo（需要先安装）
# pip install git-filter-repo
git filter-repo --path checkpoint/ --invert-paths
git filter-repo --path checkpoints/ --invert-paths

# 强制推送（危险操作，需要团队成员配合）
git push origin --force --all
git push origin --force --tags
```

**⚠️ 严重警告**：
- 这会重写Git历史
- 如果其他人已经拉取了代码，会造成冲突
- 需要所有团队成员重新克隆仓库
- 建议在个人分支或新仓库操作

### 3.3 使用 BFG Repo-Cleaner（适合大文件）

BFG是专门用于清理Git历史中大文件的工具：

```bash
# 下载BFG: https://rtyley.github.io/bfg-repo-cleaner/

# 克隆一个镜像仓库（bare repository）
git clone --mirror https://github.com/user/repo.git

# 删除checkpoint目录
java -jar bfg.jar --delete-folders checkpoint
java -jar bfg.jar --delete-folders checkpoints

# 清理
cd repo.git
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 推送
git push
```

---

## 方案四：如果文件特别大（Git LFS）

如果checkpoint文件很大，考虑使用Git LFS（Large File Storage）：

```bash
# 安装Git LFS
git lfs install

# 追踪大文件
git lfs track "*.pth"
git lfs track "*.pt"
git lfs track "*.ckpt"

# 添加.gitattributes
git add .gitattributes

# 从历史中迁移现有文件到LFS
git lfs migrate import --include="*.pth,*.pt,*.ckpt" --everything
```

---

## 推荐的最佳实践

1. **预防措施**：
   ```bash
   # 在.gitignore中确保包含：
   checkpoint/
   checkpoints/
   *.pth
   *.pt
   *.ckpt
   log/
   ```

2. **提交前检查**：
   ```bash
   git status  # 检查是否有不应该提交的文件
   git diff --cached  # 查看暂存区的更改
   ```

3. **如果已经提交但未推送**：
   - 使用 `git reset` 撤销提交
   - 从暂存区移除checkpoint文件
   - 重新提交

4. **如果已经推送**：
   - 如果是个人分支：使用 `git rm --cached` 移除跟踪
   - 如果已经合并到主分支：考虑使用BFG或git-filter-repo
   - 通知团队成员需要重新克隆或重置

---

## 快速检查命令

```bash
# 检查checkpoint文件是否被Git跟踪
git ls-files | grep -E "\.(pth|pt|ckpt)$|checkpoint/|checkpoints/"

# 检查文件大小
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '/^blob/ {print substr($0,6)}' | \
  sort --numeric-sort --key=2 | \
  tail -20

# 检查最近提交中的大文件
git log --all --pretty=format: --name-only --diff-filter=A | \
  sort -u | \
  xargs -I {} sh -c 'git log -1 --format="%H %ai {}" -- {} && git cat-file -s $(git log -1 --format="%T" -- {}) 2>/dev/null' | \
  sort -k4 -rn | head -20
```

---

## 总结

- ✅ **未推送**：使用 `git reset` 撤销提交
- ⚠️ **已推送**：使用 `git rm --cached` 停止跟踪，或使用BFG/git-filter-repo清理历史
- 🔒 **最重要**：确保 `.gitignore` 正确配置，避免未来误提交

