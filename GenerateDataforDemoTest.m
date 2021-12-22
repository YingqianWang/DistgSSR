%% Initialization
clear all;
clc;
%close all;
addpath(genpath('./Functions/'))


%% Parameters setting

angRes = 5; % Angular Resolution, options, e.g., 3, 5, 7, 9. Default: 5
factor = 4;
downRatio = 1/factor;
sourceDataPath = '../Datasets/';
sourceDatasets = dir(sourceDataPath);
sourceDatasets(1:2) = [];
datasetsNum = length(sourceDatasets);

for DatasetIndex = 1 : 5
    DatasetName = sourceDatasets(DatasetIndex).name;
    
    sourceDataFolder = [sourceDataPath, sourceDatasets(DatasetIndex).name, '/test/'];
    folders = dir(sourceDataFolder); % list the scenes
    if isempty(folders)
        continue
    end
    folders(1:2) = [];
    sceneNum = length(folders);
    
    for iScene = 1 : sceneNum
        idx_s = 0;
        sceneName = folders(iScene).name;
        sceneName(end-3:end) = [];
        fprintf('Generating test data of Scene_%s in Dataset %s......\n', sceneName, sourceDatasets(DatasetIndex).name);
        dataPath = [sourceDataFolder, folders(iScene).name];
        data = load(dataPath);
        LF = data.LF;
        [U, V, H, W, ~] = size(LF);
        while mod(H, 4) ~= 0
            H = H - 1;
        end
        while mod(W, 4) ~= 0
            W = W - 1;
        end
        
        LF = LF(0.5*(U-angRes+2):0.5*(U+angRes), 0.5*(V-angRes+2):0.5*(V+angRes), 1:H, 1:W, 1:3); % Extract central angRes*angRes views
        [U, V, H, W, ~] = size(LF);      
        
        SavePath = ['./input/', DatasetName, '_', sceneName, '/'];
        if exist(SavePath, 'dir')==0
            mkdir(SavePath);
        end
        for u = 1 : U
            for v = 1 : V
                SAI_rgb = squeeze(LF(u, v, :, :, :));
                LR_rgb = imresize(SAI_rgb, downRatio);
                imwrite(LR_rgb, [SavePath, 'view_', num2str(u, '%02d'), '_', num2str(v, '%02d'), '.png']);
            end
        end

    end
end


