## Install MPI
### Linux
```bash
sudo apt install mpich
```
### macOS
```bash
brew install mpich
```

## Build & run
```bash
cd build
cmake ..
make
./run.sh $(nb process) $(problem json file) $(result path)
```
Example:
```bash
./run.sh 4 problem.json result.txt
```