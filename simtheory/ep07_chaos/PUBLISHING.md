# SimTheory #07: publishing copy

## Title (45 characters)
Can a Computer Predict Chaos? | SimTheory #07

## Description
Three identical double-pendulum simulations: same physics, same start, same code. The only difference is how many digits the computer keeps. The 7-digit copy goes wrong after 9.7 s, the 16-digit copy after 23.1 s. Every extra digit buys only about 1.5 s more prediction, so seeing 1 minute ahead takes ~40 digits.

Real physics: identical simulations at 7, 16 and 100 digits of precision. All times measured.

Real-world measurements are far less precise than 16 digits, so a real chaotic system can't be predicted far ahead, no matter how big the computer is.

How it was measured: RK4 integrator, dt = 1 ms, both arms released from 120 deg. Runs in float32, float64 and 24/32/40/100-digit arbitrary precision (the 100-digit reference checked against 120 digits). "Wrong" = bottom arm off by more than 1 degree.

#SimTheory #ChaosTheory #DoublePendulum #Physics #Shorts

## Hashtags (5)
#SimTheory #ChaosTheory #DoublePendulum #Physics #Shorts

## Pinned comment
What should SimTheory simulate next? Drop your idea below. The most-liked one becomes Episode 8.
