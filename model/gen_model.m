raw_prefix = "../../Data/DataExtracted";
nfiles = 1; % number of files to read through (only works with 1)
n = 16; % number of muscles
g = 17; % number of gestures
filter = 100;


% gestures, muscles
gesture_data = filter * ones(g,n);

for f=1:nfiles
    fname = strcat(strcat(raw_prefix, int2str(f)), ".mat");
    raw = load(fname);
    
    cellsize = size(raw.DATA);
    for gn=1:g % gestures
        c = 0;
        for i=1:cellsize(1) % Select an individual cell
            d = raw.DATA{i,gn}; % Load data array of datapoint x muscle
            s = size(d);
            for j=1:s(1) % grab individual rows of data
                m1 = d(j,1); 
                for k=1:n % per muscle
                    ratio = m1/d(j,k);
                    
                    % filter out invalid data
                    if abs(ratio) > filter * abs(gesture_data(gn,k))
                        ratio = gesture_data(gn,k);
                    end
                    % Cumulative moving average
                    gesture_data(gn,k) = (ratio+c*gesture_data(gn,k))/(c+1); 
                    
                    % Regular average
                    %gesture_data(gn,k) = gesture_data(gn,k) + ratio;
                end
                c = c + 1;
            end
        end
        
        for k=1:n % per muscle
            %gesture_data(gn,k) = gesture_data(gn,k)/c;
        end
    end
end
gesture_data
writematrix(gesture_data, "model.csv")