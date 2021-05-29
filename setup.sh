tmp_dir='py_sparrow_rec_sys_tmp_dir'
rm -rf $tmp_dir && mkdir $tmp_dir 
cd $tmp_dir
git clone https://github.com/wzhe06/SparrowRecSys.git
cp -r SparrowRecSys/src .. && cd .. && rm $tmp_dir
