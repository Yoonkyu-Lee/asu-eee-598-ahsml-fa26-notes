#!/bin/bash
# Rebuild the CP12 Timeloop/Accelergy toolchain under /opt/tl in WSL Ubuntu 24.04 (run as root), at the commits
# recorded in cp/CP12.md. Used to set up a second machine (2026-11-01).
#   wsl -d Ubuntu-24.04 -u root --exec bash src/tl_cp12/build_wsl.sh
set -ex
export DEBIAN_FRONTEND=noninteractive GIT_TERMINAL_PROMPT=0
apt-get update -q
apt-get install -y -q scons libconfig++-dev libboost-dev libboost-iostreams-dev libboost-serialization-dev \
  libboost-thread-dev libboost-program-options-dev libboost-filesystem-dev libyaml-cpp-dev libncurses-dev \
  libtinfo-dev libgpm-dev git build-essential python3-pip python3-venv libisl-dev libntl-dev libgmp-dev cmake \
  pkg-config autoconf automake libtool wget
mkdir -p /opt/tl/runs && cd /opt/tl

clone() {  # repo url, dir, commit (empty = default branch head)
  [ -d "$2" ] || git clone -q "$1" "$2"
  if [ -n "$3" ]; then (cd "$2" && git checkout -q "$3"); fi
  echo "$2 $(cd "$2" && git log -1 --format=%h_%cs)"
}

# CACTI (master, no commit recorded) and barvinok 0.41.6
clone https://github.com/HewlettPackard/cacti.git cacti ""
(cd cacti && make -j"$(nproc)" > /tmp/cacti.log 2>&1)
if [ ! -f /usr/local/lib/libbarvinok.so ]; then
  [ -f barvinok-0.41.6.tar.gz ] || wget -q https://barvinok.sourceforge.io/barvinok-0.41.6.tar.gz
  tar xzf barvinok-0.41.6.tar.gz
  (cd barvinok-0.41.6 && ./configure --enable-shared-barvinok > /tmp/barv_conf.log 2>&1 \
     && make -j"$(nproc)" > /tmp/barv_make.log 2>&1 && make install > /tmp/barv_inst.log 2>&1)
  ldconfig
fi

# Timeloop
clone https://github.com/NVlabs/timeloop.git timeloop 3237082
(cd timeloop/src && ln -sf ../pat-public/src/pat .)
(cd timeloop && scons -j"$(nproc)" --accelergy > /tmp/tl_build.log 2>&1)
ls timeloop/bin

# Accelergy, plug-ins, timeloopfe, exercises (components)
[ -d venv ] || python3 -m venv venv
. venv/bin/activate
pip install -q --upgrade pip
clone https://github.com/Accelergy-Project/accelergy.git accelergy 6911d15
clone https://github.com/Accelergy-Project/accelergy-aladdin-plug-in.git accelergy-aladdin-plug-in 5e2e126
clone https://github.com/Accelergy-Project/accelergy-cacti-plug-in.git accelergy-cacti-plug-in 7649b2c
clone https://github.com/Accelergy-Project/accelergy-library-plug-in.git accelergy-library-plug-in ba4e9da
clone https://github.com/Accelergy-Project/timeloopfe.git timeloopfe 5603893
clone https://github.com/Accelergy-Project/timeloop-accelergy-exercises.git timeloop-accelergy-exercises 2d55108
for r in accelergy accelergy-aladdin-plug-in accelergy-library-plug-in timeloopfe; do (cd "$r" && pip install -q .); done
# the CACTI plug-in's submodule is empty: install it with the CACTI built above
(cd accelergy-cacti-plug-in && rm -rf cacti && cp -r /opt/tl/cacti cacti && pip install -q .)
P=/opt/tl/venv/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in
mkdir -p $P/cacti && cp /opt/tl/cacti/cacti $P/cacti/cacti && cp -r /opt/tl/cacti/tech_params $P/cacti/ 2>/dev/null || true
# CP12 SRAM-cell sensitivity: a copy of the plug-in with itrs-hp cells
rm -rf /opt/tl/cacti_hp_plugin && cp -r $P /opt/tl/cacti_hp_plugin
sed -i 's/itrs-lstp/itrs-hp/g' /opt/tl/cacti_hp_plugin/default_cfg.cfg
pip list 2>/dev/null | grep -i -E 'accelergy|timeloop|plug'
echo BUILD_OK
