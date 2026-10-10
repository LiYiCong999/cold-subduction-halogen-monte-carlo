# GitHub 代码发布与 Zenodo DOI 关联教程

## 预先检查

本压缩包含 3 个按论文分析逻辑顺序编号的 .py 脚本（01 Release efficiency → 02 Halogen dehydration → 03 Mélange mixing）、依赖文件和说明。三个 Python 脚本内部的注释、说明和输出提示均为英文。**尚缺实际输入 CSV**，因此只能检查 Python 语法，不能复现数值结果。论文公开前请加入有权发布的输入数据，或者明确引用公开数据的 DOI 并写出重现步骤。

**作者信息**：把 `CITATION.cff.template` 里的占位人名改成真实软件贡献者，并重命名为 `CITATION.cff`；不要仅把占位模板直接发布。请根据科研团队协议确定代码授权许可（如 MIT），由权利人添加相应 LICENSE 文件。

## A. GitHub 创建仓库并上传（网页操作，无需 Git 命令）

1. 打开 https://github.com 并登录。
2. 点击右上角 **+ → New repository**。
3. 仓库名填 `cold-subduction-halogen-monte-carlo`；简介可填 `Reproducible Monte Carlo scripts for halogen behavior during cold oceanic subduction`。建议设为 **Public**（论文公开代码时）。
4. 如果使用本包，不要初始化重复 README，以免需要手动合并。点击 **Create repository**。
5. 在电脑上**解压 ZIP**，进入解压后的仓库目录（包含 README.md、scripts、data、docs 等）。选择这个目录中的全部文件与文件夹（**不是直接选择 ZIP 文件**），拖到 GitHub 的 **uploading an existing file** 页面；已建仓库时使用 **Add file → Upload files**。检查 `scripts/` 内按 01/02/03 顺序存在全部 3 个 .py 文件，然后提交（Commit changes / Propose changes，按界面提示）。
6. 阅读仓库首页 README 是否能显示，检查 `requirements.txt`、`data/README.md` 是否都已上传。没有输入 CSV 时，不能声称代码完全可复现。

## B. 先把 GitHub 与 Zenodo 账户关联，再启用仓库

1. 登录 https://zenodo.org ，点右上角头像 → **Linked accounts**。GitHub 旁选择 **Connect**，跳转 GitHub 授权并返回 Zenodo。
2. 再点 Zenodo 头像 → **GitHub**，点击 **Sync now**（若有）。
3. 找到 `cold-subduction-halogen-monte-carlo`，打开右侧开关/滑块，确认显示为已启用（enabled）。
4. **务必在 GitHub 创建第一个 Release 前完成这一动作。**如果没找到仓库，先检查是否同一 GitHub 账户、是否需要重新同步，以及 Zenodo 是否获得授权。

## C. GitHub 创建 Release → Zenodo 获取 DOI

1. 回到 GitHub 仓库，右侧 **Releases** → **Create a new release**（或 Draft a new release）。
2. **Choose a tag** → 创建 `v1.0.0`（基于默认分支）；Release title 可写 `v1.0.0 - Initial public release`。
3. 说明填写 `Initial release of three Monte Carlo scripts supporting stage-specific release contributions, halogen dehydration and mélange mixing during cold oceanic subduction.`。
4. 如果是正式公开，确认不是 pre-release，点击 **Publish release**。
5. 回 Zenodo → 头像 → **GitHub** → 该仓库，查看 release 归档状态，完成后打开生成的 DOI 页面。自动处理不一定即时完成。
6. 论文应优先引用正式用来生成分析结果的**具体版本 DOI**。Zenodo 也有 concept DOI（所有版本的总入口）；两者用途不同。
7. 在 GitHub README 中补充实际 repo URL 和 Zenodo DOI；如使用 Zenodo DOI badge，可在 Zenodo 记录页复制官方徽章 Markdown。

## D. 你已经有 Zenodo DOI 时要特别注意

- 已发表 Zenodo **数据集**的 DOI 与 GitHub 自动归档**软件代码**的 DOI，一般是两个记录。可以分别引用，并在 Zenodo 的 **Related identifiers** 中互相关联。
- 既有的**手动上传的软件记录**不会因为连接 GitHub 就自动成为该仓库 Release 的归档历史。GitHub 自动归档通常创建新的软件记录/DOI；**不要将新 DOI 当作旧 DOI**。如果必须保持原有软件记录 DOI 系列，应先核对记录类型，再考虑在原 Zenodo 记录下新增版本和手动更新软件压缩包。
- 论文 Code availability 应明确区别代码 DOI、数据 DOI 和 GitHub 地址，不要误用文章 DOI。

## E. 常见问题

- 上传后只看见 ZIP：你上传了压缩包本身，应该先解压再将内容上传。
- Zenodo 找不到仓库：核对 GitHub 账号授权、**Sync now**、是否是仓库的 owner、私有仓库访问权限等。
- Release 没有 DOI：首先核实是否在 release 前启用了集成；然后检查 Zenodo 归档错误和 CITATION.cff 元数据。
- CSV FileNotFoundError：从项目根目录运行脚本，并把真实的 `bulk_data.csv` / `halogen_data.csv` / `release_efficiency.csv` 放到 `data/`。

## 常用链接（Zenodo 和 GitHub 官方）

- https://help.zenodo.org/docs/profile/linking-accounts/
- https://help.zenodo.org/docs/github/enable-repository/
- https://help.zenodo.org/docs/github/archive-software/github-upload/
- https://help.zenodo.org/docs/github/describe-software/citation-file/
- https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
