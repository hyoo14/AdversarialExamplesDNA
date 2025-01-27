#attack based on packbio sequencer similator(pbsim2)

# packages
!apt-get update
!apt-get install -y git g++ cmake zlib1g-dev libboost-all-dev

# git clone PBSIM2
!git clone https://github.com/yukiteruono/pbsim2.git
%cd pbsim2

# Autotools build
!autoreconf -i
!./configure
!make
!make install

# Simulate(make attack sequence)
!pbsim --prefix output_simulation \
      --hmm_model data/P6C4.model \
      --depth 1 \
      --length-min 300 --length-max 300 \
      /content/test.fasta



# save
import os

# PBSIM2 output dir
pbsim_dir = "/content/pbsim2"#"/content/build/pbsim2/build/pbsim2"
output_fasta = "/content/combined_output.fasta"

# open file
with open(output_fasta, "w") as fasta_out:
    # merge all .fastq file in the dir
    for file_name in sorted(os.listdir(pbsim_dir)):
        if file_name.endswith(".fastq"):
            file_path = os.path.join(pbsim_dir, file_name)
            with open(file_path, "r") as fastq_file:
                lines = fastq_file.readlines()
                # extract seq from FASTQ format
                for i in range(0, len(lines), 4):  # FASTQ (4 lines)
                    header = lines[i].strip()  # FASTQ header (start with @)
                    sequence = lines[i + 1].strip()  # FASTQ seq
                    # save as FASTA format
                    fasta_out.write(f">{header[1:]}\n")  # remove '@' and add '>'
                    fasta_out.write(f"{sequence}\n")

print(f"Combined FASTA file saved at: {output_fasta}")
