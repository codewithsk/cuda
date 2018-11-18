declare -a DATASETS=( "small" "ash608" "dwt_1007" "kron_g500-logn18" "rajat31" "TEM181302_M" "TEM27623_M" "wiki-Vote")
declare -a K=( "31" "32" "33" "34" "35" "64" "128" )

make clean
make

for data in "${DATASETS[@]}"
do
	for k in "${K[@]}"
	do
		./spmm_csr_driver datasets/${data}.mtx $k
	done
done
