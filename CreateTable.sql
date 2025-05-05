create schema features;
use features
create table features (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ten_file VARCHAR(255),
    zeroCrossingRate INT,
    averageEnergy FLOAT,
    averageFrequency FLOAT,
    frequencyVariation FLOAT,
    averagePitch FLOAT,
    pitchVariation FLOAT
);
select * from features
