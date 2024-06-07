%% Step 1. Obtain Datasets
% 1.1 Perform RC + CR steered plane wave transmission
% 1.2 Apply beamforming and clutter filtering to each angle to extract blood flow datasets S_RC and S_CR
% The data structures for S_RC and S_CR are as follows:
% S_RC(lateral, elevational, axial, angle, frame)
% S_CR(lateral, elevational, axial, angle, frame)

%% Step 2. Split Datasets
% K: the number of sub-datasets
for i_subsets = 1:K
    % Split RC dataset into K sub-datasets
    eval(['S_RC_Group' num2str(i_subsets) '= S_RC(:,:,:,' num2str(i_subsets) ':K:end,:);']);
    % Calculate the mean and standard deviation for each sub-dataset
    eval(['S_RC_mean{i_subsets} = mean(S_RC_Group' num2str(i_subsets) ', 4);']);
    eval(['S_RC_std{i_subsets} = std(S_RC_Group' num2str(i_subsets) ', 1, 4);']);
    % Split CR dataset into K sub-datasets
    eval(['S_CR_Group' num2str(i_subsets) '= S_CR(:,:,:,' num2str(i_subsets) ':K:end,:);']);
    % Calculate the mean and standard deviation for each sub-dataset
    eval(['S_CR_mean{i_subsets} = mean(S_CR_Group' num2str(i_subsets) ', 4);']);
    eval(['S_CR_std{i_subsets} = std(S_CR_Group' num2str(i_subsets) ', 1, 4);']);
end

%% Step 3. CCF calculation - spatial similarity
% Calculate CCF for all combinations of sub-datasets from S_CR and S_RC
for i_R = 1:K
    for i_C = 1:K
        RC_mean = squeeze(S_RC_mean{i_R});CR_mean = squeeze(S_CR_mean{i_C});
        RC_std = squeeze(S_RC_std{i_R});CR_std = squeeze(S_CR_std{i_C});
        c = ((RC_mean.*conj(CR_mean)))./...
            (abs(RC_mean).^2+abs(CR_mean).^2+(abs(RC_std).^2+abs(CR_std).^2)/2-abs(RC_mean.*conj(CR_mean)));
        c(isnan(c)) = 0;
        CCF_group{i_R, i_C} = squeeze(c);
    end
end

%% Step 4. Slow-time correlation - Temporal similarity
% Calculate and average NCCs between the calculated CCF time series
St_SW_K =  zeros(row, col, depth, 'single');
for i_R_1 = 1:K
    for i_C_1 = 1:K
        c_temp1 = gpuArray(single(CCF_group{i_R_1, i_C_1}));
        for i_R_2 = 1:K
            if i_R_1 >= i_R_2
                continue
            end
            for i_C_2 = 1:K
                if i_C_1 == i_C_2
                    continue
                end
                disp(['i_R_1: ' num2str(i_R_1) ' i_C_1:' num2str(i_C_1) ' i_R_2: ' num2str(i_R_2) ' i_C_2:' num2str(i_C_2)])
                c_temp2 = gpuArray(single(CCF_group{i_R_2, i_C_2}));
                Numetor = sum(c_temp1.*conj(c_temp2), 4);
                Denominator = sqrt((sum(c_temp1.*conj(c_temp1), 4).*sum(c_temp2.*conj(c_temp2), 4)));
                St_SW_K = St_SW_K + Numetor./Denominator;
            end
        end
    end
end

%% Step 5. Enhanced power Doppler image
% St_SW_K is the final St_SW_K weighting map
% St_SW_K = imgaussfilt3(abs(St_SW_K), 0.5); % Optional: further smooth the weighting map
St_SW_K = abs(St_SW_K/max(abs(St_SW_K(:))));
EnhancedPDImage = St_SW_K .* XDoppler; % Apply weighting map to the XDoppler image
