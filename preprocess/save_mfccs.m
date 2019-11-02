mainPath = '/home/luxb/kxw/DAVS/new_data/';
audio = 'mfcc20';

opt.fs = 16000;
opt.Tw = 25;
opt.Ts = 10;
opt.alpha = 0.97;
opt.R = [300 3700];
opt.M = 13;
opt.C = 13;
opt.L = 22;

csv_name = 'video_look_up_table.csv';
p_name = {'test', 'train','val'};
for p = 3
    c = p_name(p);    
    pathc = fullfile(mainPath, c);      % '~/val'
    for word = 1 : 500
        w = word - 1;
        pathw = fullfile(pathc, num2str(w));   % '~/val/0'
        pathw = pathw{1};               % ~/val/0
        csv_path = fullfile(pathw, csv_name);  % ~/val/0/vid/video_look_up_table.csv
        M = readtable(char(csv_path));  
        [num_files, b] = size(M);
        for k = 0:num_files-1
            file_path = fullfile(pathw, num2str(k));    % ~/val/0/0
            video_path = fullfile(file_path, [num2str(k), '.wav']);     % ~/val/0/0/0.wav
            if exist(fullfile(file_path, 'mfcc.bin'), 'file')           % ~/val//0/0/mfcc.bin
                f = fopen(fullfile(file_path, 'mfcc.bin'), 'r');
                A = fread(f, 'double');
                mfccs = reshape(A, 13, 122);
                mfccs = mfccs(2:end, :);
                for l = 2:26
                    save_mfcc20 = mfccs(:, 4 * l -7  : 4 * l + 19 -7);
                    mfcc_path = fullfile(file_path, audio);             % ~/val/0/0/mfcc20
                    if ~exist(mfcc_path, 'dir')
                        mkdir(mfcc_path)
                    end
                    f2 = fopen(fullfile(mfcc_path, [num2str(l), '.bin']), 'wb');     % ~/val/0/0/mfcc20/2.bin
                    fwrite(f2, save_mfcc20, 'double');
                    fclose(f2);                    
                end
            else
                [Speech, fs] = audioread(video_path);           % ~/val/0/0/0.wav
                Speech = Speech(1: end - 600);
                [ MFCCs, FBE, frames ] = runmfcc( Speech, opt );
                f = fopen(fullfile(file_path, 'mfcc.bin'), 'wb');   % ~/val/0/0/mfcc.bin
                fwrite(f, MFCCs, 'double');
                fclose(f);
                
                mfccs = MFCCs(2:end, :);
                for l = 2:26
                    save_mfcc20 = mfccs(:, 4 * l -7  : 4 * l + 19 -7);
                    mfcc_path = fullfile(file_path, audio);         % ~/val/0/0/mfcc20
                    if ~exist(mfcc_path, 'dir')
                        mkdir(mfcc_path)
                    end
                    f2 = fopen(fullfile(mfcc_path, [num2str(l), '.bin']), 'wb');
                    fwrite(f2, save_mfcc20, 'double');
                    fclose(f2);                    
                end
            end
        end
        
        fprintf('finish processing file num %d\n', word)
    end
end
