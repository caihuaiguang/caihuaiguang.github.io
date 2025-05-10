
# python3 jemdoc.py -c mysite_icon.conf *.jemdoc  
# # 带公式的话 
# python3 jemdoc.py -c mysite.conf  Index.jemdoc   
# python3 jemdoc_latex.py -c mysite.conf  Paper_Daily.jemdoc  
# # python2 jemdoc.py -c mysite.conf  link.jemdoc  
# # cd www
# # # python2 ../jemdoc.py  *.jemdoc  
# # python3 ../jemdoc_latex.py -c jemdoc.conf  *.jemdoc  
# # cd ..
# cd known/Blog 
# python3 ../../jemdoc.py -c mysite.conf *.jemdoc    


# python2 jemdoc.py -c mysite.conf *.jemdoc  
# cd www
# python2 ../jemdoc.py -c ../mysite.conf *.jemdoc  
# cd ..
# cd Blog 
# python2 ../jemdoc.py *.jemdoc  

python3 jemdoc_latex.py -c mysite_icon.conf *.jemdoc  
python3 jemdoc_latex.py -c mysite.conf  index.jemdoc    
cd true/courage/time/ 
python3 ../../../jemdoc_latex.py -c mysite.conf *.jemdoc  
cd ../../../www
python3 ../jemdoc_latex.py *.jemdoc  
        
# # Set the directory where your Jemdoc files are located
# jemdoc_directory="./"
# # Loop through all Jemdoc files in the directory
# for source_file in "$jemdoc_directory"/*.jemdoc; do
#     # Extract the file name without the directory path
#     file_name=$(basename "$source_file")

#     # Construct the output HTML file name based on the Jemdoc file name
#     output_file="$jemdoc_directory/${file_name%.jemdoc}.html"

#     # Check if the source file has been modified
#     if [ "$source_file" -nt "$output_file" ]; then
#         echo "Compiling $source_file to $output_file" 
#         python3 jemdoc_latex.py -c mysite_icon.conf "$source_file"  
#     else
#         echo "$source_file has not been modified. Skipping compilation."
#     fi
# done 

# python3 jemdoc_latex.py -c mysite.conf  Index.jemdoc    
# jemdoc_directory="known/Blog"
# # Loop through all Jemdoc files in the directory
# for source_file in "$jemdoc_directory"/*.jemdoc; do
#     # Extract the file name without the directory path
#     file_name=$(basename "$source_file")

#     # Construct the output HTML file name based on the Jemdoc file name
#     output_file="$jemdoc_directory/${file_name%.jemdoc}.html"

#     # Check if the source file has been modified
#     if [ "$source_file" -nt "$output_file" ]; then
#         echo "Compiling $source_file to $output_file" 
#         python3 jemdoc_latex.py -c mysite_icon.conf "$source_file"  
#     else
#         echo "$source_file has not been modified. Skipping compilation."
#     fi
# done  
