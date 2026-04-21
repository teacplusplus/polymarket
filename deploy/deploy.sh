export CARGO_BUILD_JOBS=4
export CC=gcc-10
export CXX=g++-10
cargo build --profile release
cp ../target/release/poly ./poly
rsync -avz ./ root@204.13.237.94:/home/poly
