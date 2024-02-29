

% Load the .mat file containing the 'data' structure
load('./data/cell3005.mat')

% Extract fields from the data structure
spikes = data.spikes;
target_direction = data.target_direction;
unique_directions = unique(target_direction);

% Determine the number of unique directions to figure out subplot sizing
numDirections = length(unique_directions);

% Prepare the figure
figure;

% Iterate over each unique direction
for i = 1:2%numDirections
    % Find trials corresponding to the current direction
    trialsForCurrentDirection = target_direction == unique_directions(i);
    
    % Extract spikes for the current direction
    spikesForCurrentDirection = spikes(:, trialsForCurrentDirection);
    
    % Transpose the matrix for correct orientation
    spikesForCurrentDirection = spikesForCurrentDirection';
    
    % Create a subplot for the current direction
    subplot(ceil(sqrt(numDirections)), ceil(numDirections/ceil(sqrt(numDirections))), i);
    spy(spikesForCurrentDirection, 'k', 1);
    
    % Set title for the subplot indicating the direction
    title(['Direction: ' num2str(unique_directions(i)) ' degrees']);
    
    % Adjusting axes labels for clarity
    xlabel('Time (ms)');
    ylabel('Trial');
end

% Improve layout
sgtitle('Raster plots for each target motion direction');
