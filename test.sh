for file in `ls images`
do
  echo $file
  python main.py images/$file
done
    
