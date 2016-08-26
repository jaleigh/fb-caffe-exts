
echo
echo This script will install fblualib and all its dependencies.
echo It has been Ubuntu 16.04, Linux x86_64.
echo

set -e
set -x


if [[ $(arch) != 'x86_64' ]]; then
    echo "x86_64 required" >&2
    exit 1
fi

issue=$(cat /etc/issue)
extra_packages=libiberty-dev
current=1

dir=$(mktemp --tmpdir -d fblualib-build.XXXXXX)

echo Working in $dir
echo
cd $dir

echo Installing required packages
echo
sudo apt-get install -y \
    git \
    curl \
    wget \
    g++ \
    automake \
    autoconf \
    autoconf-archive \
    libtool \
    libboost-all-dev \
    libevent-dev \
    libdouble-conversion-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    liblz4-dev \
    liblzma-dev \
    libsnappy-dev \
    make \
    zlib1g-dev \
    binutils-dev \
    libjemalloc-dev \
    $extra_packages \
    flex \
    bison \
    libkrb5-dev \
    libsasl2-dev \
    libnuma-dev \
    pkg-config \
    libssl-dev \
    libedit-dev \
    libmatio-dev \
    libpython-dev \
    libpython3-dev \
    python-numpy

echo
echo Cloning repositories
echo


git clone --depth 1 https://github.com/zhanghang1989/folly.git
git clone --depth 1 https://github.com/zhanghang1989/fbthrift
git clone https://github.com/facebook/thpp
git clone https://github.com/zhanghang1989/fblualib
git clone https://github.com/facebook/wangle

echo
echo Building folly
echo

cd $dir/folly/folly
autoreconf -ivf
./configure
make -j24
sudo make install
sudo ldconfig

if [ $current -eq 1 ]; then
    echo
    echo Wangle
    echo

    cd $dir/wangle/wangle
    git checkout d89a94139e57673abd32a78e9f371f4badbab36c
    cmake .
    make -j24
    sudo make install
fi

echo
echo Building fbthrift
echo

cd $dir/fbthrift/thrift
autoreconf -ivf
./configure
make -j24
sudo make install

echo
echo 'Installing TH++'
echo

cd $dir/thpp/thpp
./build.sh

echo
echo 'Installing FBLuaLib'
echo

cd $dir/fblualib/fblualib
./build.sh
cd $dir/fblualib/fblualib/python
luarocks make rockspec/fbpython-0.1-1.rockspec

echo
echo 'Almost done!'
echo

git clone https://github.com/torch/nn && ( cd nn && git checkout getParamsByDevice && luarocks make -j24 rocks/nn-scm-1.rockspec )

git clone https://github.com/facebook/fbtorch.git && ( cd fbtorch && luarocks make -j24 rocks/fbtorch-scm-1.rockspec )

git clone https://github.com/facebook/fbnn.git && ( cd fbnn && luarocks make -j24 rocks/fbnn-scm-1.rockspec )


echo
echo 'All done!'
echo
