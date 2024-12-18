# Ridge Comparison

Main Reference: https://github.com/desa-lab/EEG-Image-Reconstruction
The goal of these programs is test a hypothesis that one step in the reconstruction process could be improved.  
Right now, the first training step taken with the EEG data is to fit a Ridge regression to the EEG samples, predicting the VDVAE latents that should be used to construct the image that the subject is looking at. c.f. https://github.com/desa-lab/EEG-Image-Reconstruction/blob/main/thingseeg2_scripts/train_regression.py
We test many variations of other models that could be used in place of the Ridge regression. 
In all examples, training stops when 10 consecutive epochs fail to yield an improvement in distance and correlation over prior epochs (measured by adding % change in distance to % change in correlation).

The comparisons are all numbered using a 3-digit code:
- The first digit is the loss function:
    0: Just MSELoss.
    1: MSELoss + &#x2113;<sup>1</sup> size of the weights
    2: MSELoss + &#x2113;<sup>2</sup> size of the weights
- The second digit is the model:
    0: Single Linear layer of size 680 &#x00d7; 91168
    1: Two Linear layers, first 680 &#x00d7; 680, second 680 &#x00d7; 91168, with ReLU between.
    2: Three Linear layers, first and second 680 &#x00d7; 680, third 680 &#x00d7; 91168, with ReLU between each pair.
    3: One Linear layer of size 680 &#x00d7; 91168, prefilled with coefficients calculated from Ridge regression, then trained from that starting point.
    4: Same as 1, but with the 680 &#x00d7; 91168 linear layer prefilled with Ridge coefficients.
    5: Same as 2, but with the 680 &#x00d7; 91168 linear layer prefilled with Ridge coefficients.
    6: Same as 4, but with the Ridge coefficients locked so they are not trained.
    7: Same as 5, but with the Ridge coefficients locked so they are not trained.
    <!-- 8: A single 1-D convolution, length 3, stride 1, zero-padding 1, 17 EEG channels as channels, 64 output channels.   -->
- The third digit is the learning rate:
    0: Constant learning rate of 0.01
    1: Constant learning rate of 0.001
    2: Constant learning rate of 0.0001
    3: Learning rate starts at 0.01 and then decays by 1% per epoch.
    4: Learning rate starts at 0.001 and then decays by 1% per epoch.
    5: Learning rate starts at 0.0001 and then decays by 1% per epoch.

## Results

**Ridge regression results**: MSE: 113.3732, Avg Pearson Corr Coeffs: 0.023811

### Loss 0 (MSE only)

<table>
	<tr>
		<td></td>
		<td colspan='6'><center><strong>Learning Rate</strong></center></td>
	</tr>
	<tr>
		<td></td>
		<td>0.01</td>
		<td>0.001</td>
		<td>0.0001</td>
		<td>0.01<br/>with decay</td>
		<td>0.001<br/>with decay</td>
		<td>0.0001<br/>with decay</td>
	<tr> <!-- 00x -->
		<td><strong>Model 0</strong></td>
		<td><span style='color:red'>dist = 113.69</span><br/><span style='color:red'>corr = 0.01247</td><!-- 000 -->
		<td><span style='color:red'>dist = 113.63</span><br/><span style='color:red'>corr = 0.01892</td><!-- 001 -->
		<td><span style='color:red'>dist = 113.41</span><br/><span style='color:red'>corr = 0.02340</td><!-- 002 -->
		<td><span style='color:red'>dist = 113.44</span><br/><span style='color:red'>corr = 0.02308</td><!-- 003 -->
		<td><span style='color:red'>dist = 113.44</span><br/><span style='color:red'>corr = 0.02311</td><!-- 004 -->
		<td><span style='color:red'>dist = 113.40</span><br/><span style='color:red'>corr = 0.02357</td><!-- 005 -->
	</tr>
	<tr> <!-- 01x -->
		<td><strong>Model 1</strong></td>
		<td><span style='color:green'>dist = 107.26</span><br/><span style='color:red'>corr = 0.00923</td><!-- 010 -->
		<td><span style='color:green'>dist = 98.47</span><br/><span style='color:red'>corr = 0.02268</td><!-- 011 -->
		<td><span style='color:green'>dist = 112.84</span><br/><span style='color:red'>corr = 0.02289</td><!-- 012 -->
		<td><span style='color:green'>dist = 95.72</span><br/><span style='color:green'>corr = 0.03457</td><!-- 013 -->
		<td><span style='color:green'>dist = 98.56</span><br/><span style='color:green'>corr = 0.02435</td><!-- 014 -->
		<td><span style='color:green'>dist = 112.85</span><br/><span style='color:red'>corr = 0.02307</td><!-- 015 -->
	</tr>
	<tr> <!-- 02x -->
		<td><strong>Model 2</strong></td>
		<td></td>
		<td><span style='color:green'>dist = 86.61</span><br/><span style='color:green'>corr = 0.04081</td><!-- 021 -->
		<td><span style='color:green'>dist = 112.33</span><br/><span style='color:red'>corr = 0.01559</td><!-- 022 -->
		<td></td>
		<td><span style='color:green'>dist = 86.61</span><br/><span style='color:green'>corr = 0.04051</td><!-- 024 -->
		<td><span style='color:green'>dist = 112.34</span><br/><span style='color:red'>corr = 0.01566</td><!-- 025 -->
	</tr>
	<tr> <!-- 03x -->
		<td><strong>Model 3</strong></td>
		<td><span style='color:red'>dist = 113.37</span><br/><span style='color:red'>corr = 0.02381</td><!-- 030 -->
		<td><span style='color:red'>dist = 113.37</span><br/><span style='color:red'>corr = 0.02381</td><!-- 031 -->
		<td><span style='color:red'>dist = 113.37</span><br/><span style='color:red'>corr = 0.02381</td><!-- 032 -->
		<td><span style='color:red'>dist = 113.37</span><br/><span style='color:red'>corr = 0.02381</td><!-- 033 -->
		<td><span style='color:red'>dist = 113.37</span><br/><span style='color:red'>corr = 0.02381</td><!-- 034 -->
		<td><span style='color:red'>dist = 113.37</span><br/><span style='color:red'>corr = 0.02381</td><!-- 035 -->
	</tr>
	<tr> <!-- 04x -->
		<td><strong>Model 4</strong></td>
		<td><span style='color:green'>dist = 101.39</span><br/><span style='color:green'>corr = 0.02874</td><!-- 040 -->
		<td><span style='color:green'>dist = 98.28</span><br/><span style='color:green'>corr = 0.02575</td><!-- 041 -->
		<td><span style='color:green'>dist = 113.22</span><br/><span style='color:red'>corr = 0.02314</td><!-- 042 -->
		<td><span style='color:green'>dist = 112.06</span><br/><span style='color:red'>corr = 0.00985</td><!-- 043 -->
		<td><span style='color:green'>dist = 97.99</span><br/><span style='color:green'>corr = 0.02957</td><!-- 044 -->
		<td><span style='color:green'>dist = 113.22</span><br/><span style='color:red'>corr = 0.02310</td><!-- 045 -->
	</tr>
	<tr> <!-- 05x -->
		<td><strong>Model 5</strong></td>
		<td><span style='color:green'>dist = 86.60</span><br/><span style='color:green'>corr = 0.04098</td><!-- 050 -->
		<td><span style='color:green'>dist = 89.43</span><br/><span style='color:green'>corr = 0.03797</td><!-- 051 -->
		<td><span style='color:green'>dist = 112.24</span><br/><span style='color:red'>corr = 0.02254</td><!-- 052 -->
		<td><span style='color:green'>dist = 86.60</span><br/><span style='color:green'>corr = 0.04099</td><!-- 053 -->
		<td><span style='color:green'>dist = 88.59</span><br/><span style='color:green'>corr = 0.04113</td><!-- 054 -->
		<td><span style='color:green'>dist = 112.24</span><br/><span style='color:red'>corr = 0.02251</td><!-- 055 -->
	</tr>
	<tr> <!-- 06x -->
		<td><strong>Model 6</strong></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
	</tr>
	<tr> <!-- 07x -->
		<td><strong>Model 7</strong></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
	</tr>
</table>

### Loss 1 (MSE only + &#x2113;<sup>1</sup> on weights)

<table>
	<tr>
		<td></td>
		<td colspan='6'><center><strong>Learning Rate</strong></center></td>
	</tr>
	<tr>
		<td></td>
		<td>0.01</td>
		<td>0.001</td>
		<td>0.0001</td>
		<td>0.01<br/>with decay</td>
		<td>0.001<br/>with decay</td>
		<td>0.0001<br/>with decay</td>
	<tr> <!-- 10x -->
		<td><strong>Model 0</strong></td>
		<td><span style='color:red'>dist = 114.05</span><br/><span style='color:red'>corr = 0.00922</td><!-- 100 -->
		<td><span style='color:red'>dist = 113.82</span><br/><span style='color:red'>corr = 0.01370</td><!-- 101 -->
		<td><span style='color:red'>dist = 113.49</span><br/><span style='color:red'>corr = 0.01930</td><!-- 102 -->
		<td><span style='color:red'>dist = 113.44</span><br/><span style='color:red'>corr = 0.01882</td><!-- 103 -->
		<td><span style='color:red'>dist = 113.82</span><br/><span style='color:red'>corr = 0.01370</td><!-- 104 -->
		<td><span style='color:red'>dist = 113.47</span><br/><span style='color:red'>corr = 0.01935</td><!-- 105 -->
	</tr>
	<tr> <!-- 11x -->
		<td><strong>Model 1</strong></td>
		<td><span style='color:green'>dist = 109.26</span><br/><span style='color:red'>corr = 0.00913</td><!-- 110 -->
		<td><span style='color:green'>dist = 111.65</span><br/><span style='color:red'>corr = 0.02275</td><!-- 111 -->
		<td><span style='color:green'>dist = 111.60</span><br/><span style='color:red'>corr = 0.01137</td><!-- 112 -->
		<td><span style='color:green'>dist = 109.19</span><br/><span style='color:red'>corr = 0.00994</td><!-- 113 -->
		<td><span style='color:green'>dist = 111.70</span><br/><span style='color:green'>corr = 0.02649</td><!-- 114 -->
		<td><span style='color:green'>dist = 111.61</span><br/><span style='color:red'>corr = 0.01136</td><!-- 115 -->
	</tr>
	<tr> <!-- 12x -->
		<td><strong>Model 2</strong></td>
		<td><span style='color:green'>dist = 109.49</span><br/><span style='color:red'>corr = 0.00711</td><!-- 120 -->
		<td><span style='color:green'>dist = 110.73</span><br/><span style='color:red'>corr = 0.02130</td><!-- 121 -->
		<td><span style='color:green'>dist = 109.24</span><br/><span style='color:red'>corr = 0.00984</td><!-- 122 -->
		<td><span style='color:green'>dist = 109.32</span><br/><span style='color:red'>corr = 0.00743</td><!-- 123 -->
		<td><span style='color:green'>dist = 110.73</span><br/><span style='color:red'>corr = 0.02188</td><!-- 124 -->
		<td><span style='color:green'>dist = 109.26</span><br/><span style='color:red'>corr = 0.00950</td><!-- 125 -->
	</tr>
	<tr> <!-- 13x -->
		<td><strong>Model 3</strong></td>
		<td><span style='color:red'>dist = 113.37</span><br/><span style='color:red'>corr = 0.02381</td><!-- 130 -->
		<td><span style='color:red'>dist = 113.37</span><br/><span style='color:red'>corr = 0.02381</td><!-- 131 -->
		<td><span style='color:green'>dist = 113.35</span><br/><span style='color:red'>corr = 0.02381</td><!-- 132 -->
		<td><span style='color:red'>dist = 113.37</span><br/><span style='color:red'>corr = 0.02381</td><!-- 133 -->
		<td><span style='color:red'>dist = 113.37</span><br/><span style='color:red'>corr = 0.02381</td><!-- 134 -->
		<td><span style='color:green'>dist = 113.35</span><br/><span style='color:red'>corr = 0.02381</td><!-- 135 -->
	</tr>
	<tr> <!-- 14x -->
		<td><strong>Model 4</strong></td>
		<td><span style='color:green'>dist = 109.83</span><br/><span style='color:red'>corr = 0.01019</td><!-- 140 -->
		<td><span style='color:green'>dist = 111.56</span><br/><span style='color:red'>corr = 0.02239</td><!-- 141 -->
		<td><span style='color:green'>dist = 111.43</span><br/><span style='color:red'>corr = 0.01767</td><!-- 142 -->
		<td><span style='color:green'>dist = 109.70</span><br/><span style='color:red'>corr = 0.00911</td><!-- 143 -->
		<td><span style='color:green'>dist = 111.69</span><br/><span style='color:green'>corr = 0.02511</td><!-- 144 -->
		<td><span style='color:green'>dist = 111.42</span><br/><span style='color:red'>corr = 0.01778</td><!-- 145 -->
	</tr>
	<tr> <!-- 15x -->
		<td><strong>Model 5</strong></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
	</tr>
	<tr> <!-- 16x -->
		<td><strong>Model 6</strong></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
	</tr>
	<tr> <!-- 17x -->
		<td><strong>Model 7</strong></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
	</tr>
</table>

### Loss 2 (MSE only + &#x2113;<sup>2</sup> on weights)

<table>
	<tr>
		<td></td>
		<td colspan='6'><center><strong>Learning Rate</strong></center></td>
	</tr>
	<tr>
		<td></td>
		<td>0.01</td>
		<td>0.001</td>
		<td>0.0001</td>
		<td>0.01<br/>with decay</td>
		<td>0.001<br/>with decay</td>
		<td>0.0001<br/>with decay</td>
	<tr> <!-- 20x -->
		<td><strong>Model 0</strong></td>
		<td><span style='color:red'>dist = 114.13</span><br/><span style='color:red'>corr = 0.00569</td><!-- 200 -->
		<td><span style='color:red'>dist = 114.02</span><br/><span style='color:red'>corr = 0.00765</td><!-- 201 -->
		<td><span style='color:red'>dist = 113.61</span><br/><span style='color:red'>corr = 0.01460</td><!-- 202 -->
		<td><span style='color:red'>dist = 114.11</span><br/><span style='color:red'>corr = 0.00601</td><!-- 203 -->
		<td><span style='color:red'>dist = 114.02</span><br/><span style='color:red'>corr = 0.00765</td><!-- 204 -->
		<td><span style='color:red'>dist = 113.61</span><br/><span style='color:red'>corr = 0.01463</td><!-- 205 -->
	</tr>
	<tr> <!-- 21x -->
		<td><strong>Model 1</strong></td>
		<td></td>
		<td><span style='color:green'>dist = 111.76</span><br/><span style='color:red'>corr = 0.00493</td><!-- 211 -->
		<td><span style='color:green'>dist = 108.68</span><br/><span style='color:red'>corr = 0.00536</td><!-- 212 -->
		<td></td>
		<td><span style='color:green'>dist = 111.76</span><br/><span style='color:red'>corr = 0.00493</td><!-- 214 -->
		<td><span style='color:green'>dist = 108.68</span><br/><span style='color:red'>corr = 0.00537</td><!-- 215 -->
	</tr>
	<tr> <!-- 22x -->
		<td><strong>Model 2</strong></td>
		<td></td>
		<td></td>
		<td><span style='color:green'>dist = 112.84</span><br/><span style='color:red'>corr = 0.00451</td><!-- 222 -->
		<td></td>
		<td></td>
		<td><span style='color:green'>dist = 112.90</span><br/><span style='color:red'>corr = 0.00451</td><!-- 225 -->
	</tr>
	<tr> <!-- 23x -->
		<td><strong>Model 3</strong></td>
		<td><span style='color:red'>dist = 113.37</span><br/><span style='color:red'>corr = 0.02381</td><!-- 230 -->
		<td><span style='color:red'>dist = 113.37</span><br/><span style='color:red'>corr = 0.02381</td><!-- 231 -->
		<td><span style='color:red'>dist = 113.37</span><br/><span style='color:red'>corr = 0.02381</td><!-- 232 -->
		<td><span style='color:red'>dist = 113.37</span><br/><span style='color:red'>corr = 0.02381</td><!-- 233 -->
		<td><span style='color:red'>dist = 113.37</span><br/><span style='color:red'>corr = 0.02381</td><!-- 234 -->
		<td><span style='color:red'>dist = 113.37</span><br/><span style='color:red'>corr = 0.02381</td><!-- 235 -->
	</tr>
	<tr> <!-- 24x -->
		<td><strong>Model 4</strong></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
	</tr>
	<tr> <!-- 25x -->
		<td><strong>Model 5</strong></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
	</tr>
	<tr> <!-- 26x -->
		<td><strong>Model 6</strong></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
	</tr>
	<tr> <!-- 27x -->
		<td><strong>Model 7</strong></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
	</tr>
</table>
