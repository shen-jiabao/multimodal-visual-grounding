运维脚本归档（赛程期间云端实例 /root 镜像）
============================================

来源：本队赛程期间使用的云端 GPU 实例的 `/root` 目录镜像（工作区
`d:\Desktop\2026AiC\semifinal_remote_archive\root\`）。
本目录为其**脚本子集**（`*.sh` + `*.py`，共 246 个文件，约 0.61 MB），
逐文件 SHA256 见同目录 `_清单_SHA256.txt`。

用途：证明 Qwen3.8-27B 环境"从零到起服"是**本队自己一步步调通的**——
下面是这批脚本所对应的真实过程（脚本名即过程本身）。

一、环境构建与权重下载
----------------------
  _q38_env.sh                环境准备与检查项
  _q38_build.sh / _q38_build_launch.sh / _q38_build_poll.sh / _q38_buildstat.sh
                             独立 venv（venv_q38）构建与进度
  _q38_dl.sh / _q38_dl_launch.sh / _q38_dl_poll.sh / _q38_dl_diag.sh
                             18 分片权重下载：断点续传 + 跨分片并发 + 逐片 sha256 双验
  _q38_weights_verify.sh     权重完整性复核
  _q38_fixsmall.sh           小文件静默截断的处置（实测坑）

二、cu130 + A100 组合下的依赖诊断（多轮试错）
--------------------------------------------
  _q38_findnvcc.sh           定位 nvcc（flashinfer JIT 需要）
  _q38_cccl.sh / _q38_cccl2.sh / _q38_cccl3.sh    CCCL 依赖排查
  _q38_lib64.sh / _q38_sofix.sh                   libX.so → libX.so.N 符号链接修复
  _q38_shm.sh                                     /dev/shm 排查（排除共享内存假设）
  _q38_compat.sh / _q38_diag.sh / _q38_check2.sh  兼容性与环境诊断
  _q38_fi.sh … _q38_fi5.sh                        flashinfer 后端逐轮试错

三、起服与就绪自证
------------------
  _get_serve_q38.sh          ★交付件级起服脚本（目录断言 / TP 判定 / 权重完整性 /
                            flashinfer 修复 / --disable-custom-all-reduce / 逐后端重试）
  _q38_autoserve.sh / _q38_readserve.sh / _q38_poll.sh / _serve_launch.sh / _serve_poll.sh
                            起服与就绪轮询（以 HTTP 为准）
  _q38_knob.sh / _q38_cfg_dump.sh                起服参数与配置回读

四、闸门与全量发车
------------------
  _q38_gate.py / _q38_gate.sh    闸门 G2（机检可解析率）+ G3（目检图）实现
  _launch_q38.sh                 全量启动器：五道闸门全过才发车
  _q38_next.sh / _q38_wait_done.sh / _q38_prog.sh / _gate_detect_test.sh

五、产物保留与关服归档
----------------------
  _env_record_q38.sh / _new_env_record.sh   环境实录采集（→ env_record.txt）
  _new_snap.sh                              原子快照（避免半截文件被取走）
  _q38_preserve.sh / _q38_preserve_run.sh   产物保留
  _q38_kill_dupmon.sh                       防双进程争卡
  _final_shutdown.sh / _archive_remote.sh   关服前归档

六、同源佐证（本目录为何可信）
------------------------------
  下列脚本的 SHA256 同时存在于三处，**逐位相同**：
    · 本归档（本队工作区）
    · 43504 实例的 `/root/rivermind-data/`（见 `../远端文件清单_SHA256.txt`）
    · 随包 `../远端原件脚本/`

    _get_serve_q38.sh  409DCD334C59181EEFD163A645A0FD8639359477234C97D198314D062430AAC9
    _q38_gate.py       3CADA92B2668A44F9864C9E85B24F24F673580941911E479B4670B102D2764FD
    _q38_gate.sh       28B83CD2E799370FD07F4C7378FE035EDE0AAFAAADF3343304582A4CB325C910
    _launch_q38.sh     C6E7F5AB401EA65960A2791EB4ACE9833EFD646E31CF8260C0FA6F48C3EEB48B
    _q38_dl.sh         1404E58A44B70D94DBAFF9F46AE2AED2473B965E34920B1802E81AC078794333
    _q38_build.sh      2A6E997C0406168868D001CDC29D52F3DC66BFC6E1E4DA047E0A5BDA12CCD055

  ⇒ 说明本队在不同实例上跑的是**同一套自研工具链**（不是逐机重写）。

七、如实说明
------------
  · 本归档中还包含对**其它开源模型**的可行性探测脚本（如 `_check_internvl.sh`、
    `_dl_235b.sh`、`_get_serve_qwen235.sh`、`_m_*` 等）——均属赛程中的探索，
    **未用于本交付件**；保留是为如实记录过程，不代表采用了这些模型。
  · 全部脚本的路径为远端绝对路径（`/root/rivermind-data`）；随包 `code/remote_shell/`
    下的同名脚本是其**原样副本**，`code/*.py` 才做了"路径可移植化"。
  · 已核查：本归档及随包所有文本**不含任何明文凭据**（凭据一律只从环境变量读取）。
