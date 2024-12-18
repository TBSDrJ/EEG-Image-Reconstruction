# Ridge Comparison

Main Reference: https://github.com/desa-lab/EEG-Image-Reconstruction
The goal of these programs is test a hypothesis that one step in the reconstruction process could be improved.  
Right now, the first training step taken with the EEG data is to fit a Ridge regression to the EEG samples, predicting the VDVAE latents that should be used to construct the image that the subject is looking at. c.f. https://github.com/desa-lab/EEG-Image-Reconstruction/blob/main/thingseeg2_scripts/train_regression.py
We test many variations of other models that could be used in place of the Ridge regression. 
In all examples, training stops when 10 consecutive epochs fail to yield an improvement in distance and correlation over prior epochs (measured by adding % change in distance to % change in correlation).

The comparisons are all numbered using a 3-digit code:
- The first digit is the loss function:
    - 0: Just MSELoss.
    - 1: MSELoss + &#x2113;<sup>1</sup> size of the weights
    - 2: MSELoss + &#x2113;<sup>2</sup> size of the weights
- The second digit is the model:
    - 0: Single Linear layer of size 680 &#x00d7; 91168
    - 1: Two Linear layers, first 680 &#x00d7; 680, second 680 &#x00d7; 91168, with ReLU between.
    - 2: Three Linear layers, first and second 680 &#x00d7; 680, third 680 &#x00d7; 91168, with ReLU between each pair.
    - 3: One Linear layer of size 680 &#x00d7; 91168, prefilled with coefficients calculated from Ridge regression, then trained from that starting point.
    - 4: Same as 1, but with the 680 &#x00d7; 91168 linear layer prefilled with Ridge coefficients.
    - 5: Same as 2, but with the 680 &#x00d7; 91168 linear layer prefilled with Ridge coefficients.
    - 6: Same as 4, but with the Ridge coefficients locked so they are not trained.
    - 7: Same as 5, but with the Ridge coefficients locked so they are not trained.
	- 8+: I plan to try a couple of models that start with one or more 1D convolutions.
    <!-- 8: A single 1-D convolution, length 3, stride 1, zero-padding 1, 17 EEG channels as channels, 64 output channels.   -->
- The third digit is the learning rate:
    - 0: Constant learning rate of 0.01
    - 1: Constant learning rate of 0.001
    - 2: Constant learning rate of 0.0001
    - 3: Learning rate starts at 0.01 and then decays by 1% per epoch.
    - 4: Learning rate starts at 0.001 and then decays by 1% per epoch.
    - 5: Learning rate starts at 0.0001 and then decays by 1% per epoch.

## Results

**Ridge regression results**: MSE: 113.3732, Avg Pearson Corr Coeffs: 0.023811

Values below in bold are better than Ridge, others are not better.

Individual missing values are models that gave `nan` values for some reason; I will try to re-run these on CUDA to see if it is an MPS-specific problem (or maybe MacOS 13-specific problem). Entire missing rows are for code I haven't built/run yet.

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
		<td>dist = 113.69<br/>corr = 0.01247</td><!-- 000 -->
		<td>dist = 113.63<br/>corr = 0.01892</td><!-- 001 -->
		<td>dist = 113.41<br/>corr = 0.02340</td><!-- 002 -->
		<td>dist = 113.44<br/>corr = 0.02308</td><!-- 003 -->
		<td>dist = 113.44<br/>corr = 0.02311</td><!-- 004 -->
		<td>dist = 113.40<br/>corr = 0.02357</td><!-- 005 -->
	</tr>
	<tr> <!-- 01x -->
		<td><strong>Model 1</strong></td>
		<td><strong>dist = 107.26</strong><br/>corr = 0.00923</td><!-- 010 -->
		<td><strong>dist = 98.47</strong><br/>corr = 0.02268</td><!-- 011 -->
		<td><strong>dist = 112.84</strong><br/>corr = 0.02289</td><!-- 012 -->
		<td><strong>dist = 95.72</strong><br/><strong>corr = 0.03457</strong></td><!-- 013 -->
		<td><strong>dist = 98.56</strong><br/><strong>corr = 0.02435</strong></td><!-- 014 -->
		<td><strong>dist = 112.85</strong><br/>corr = 0.02307</td><!-- 015 -->
	</tr>
	<tr> <!-- 02x -->
		<td><strong>Model 2</strong></td>
		<td></td>
		<td><strong>dist = 86.61</strong><br/><strong>corr = 0.04081</strong></td><!-- 021 -->
		<td><strong>dist = 112.33</strong><br/>corr = 0.01559</td><!-- 022 -->
		<td></td>
		<td><strong>dist = 86.61</strong><br/><strong>corr = 0.04051</strong></td><!-- 024 -->
		<td><strong>dist = 112.34</strong><br/>corr = 0.01566</td><!-- 025 -->
	</tr>
	<tr> <!-- 03x -->
		<td><strong>Model 3</strong></td>
		<td>dist = 113.37<br/>corr = 0.02381</td><!-- 030 -->
		<td>dist = 113.37<br/>corr = 0.02381</td><!-- 031 -->
		<td>dist = 113.37<br/>corr = 0.02381</td><!-- 032 -->
		<td>dist = 113.37<br/>corr = 0.02381</td><!-- 033 -->
		<td>dist = 113.37<br/>corr = 0.02381</td><!-- 034 -->
		<td>dist = 113.37<br/>corr = 0.02381</td><!-- 035 -->
	</tr>
	<tr> <!-- 04x -->
		<td><strong>Model 4</strong></td>
		<td><strong>dist = 101.39</strong><br/><strong>corr = 0.02874</strong></td><!-- 040 -->
		<td><strong>dist = 98.28</strong><br/><strong>corr = 0.02575</strong></td><!-- 041 -->
		<td><strong>dist = 113.22</strong><br/>corr = 0.02314</td><!-- 042 -->
		<td><strong>dist = 112.06</strong><br/>corr = 0.00985</td><!-- 043 -->
		<td><strong>dist = 97.99</strong><br/><strong>corr = 0.02957</strong></td><!-- 044 -->
		<td><strong>dist = 113.22</strong><br/>corr = 0.02310</td><!-- 045 -->
	</tr>
	<tr> <!-- 05x -->
		<td><strong>Model 5</strong></td>
		<td><strong>dist = 86.60</strong><br/><strong>corr = 0.04098</strong></td><!-- 050 -->
		<td><strong>dist = 89.43</strong><br/><strong>corr = 0.03797</strong></td><!-- 051 -->
		<td><strong>dist = 112.24</strong><br/>corr = 0.02254</td><!-- 052 -->
		<td><strong>dist = 86.60</strong><br/><strong>corr = 0.04099</strong></td><!-- 053 -->
		<td><strong>dist = 88.59</strong><br/><strong>corr = 0.04113</strong></td><!-- 054 -->
		<td><strong>dist = 112.24</strong><br/>corr = 0.02251</td><!-- 055 -->
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
		<td>dist = 114.05<br/>corr = 0.00922</td><!-- 100 -->
		<td>dist = 113.82<br/>corr = 0.01370</td><!-- 101 -->
		<td>dist = 113.49<br/>corr = 0.01930</td><!-- 102 -->
		<td>dist = 113.44<br/>corr = 0.01882</td><!-- 103 -->
		<td>dist = 113.82<br/>corr = 0.01370</td><!-- 104 -->
		<td>dist = 113.47<br/>corr = 0.01935</td><!-- 105 -->
	</tr>
	<tr> <!-- 11x -->
		<td><strong>Model 1</strong></td>
		<td><strong>dist = 109.26</strong><br/>corr = 0.00913</td><!-- 110 -->
		<td><strong>dist = 111.65</strong><br/>corr = 0.02275</td><!-- 111 -->
		<td><strong>dist = 111.60</strong><br/>corr = 0.01137</td><!-- 112 -->
		<td><strong>dist = 109.19</strong><br/>corr = 0.00994</td><!-- 113 -->
		<td><strong>dist = 111.70</strong><br/><strong>corr = 0.02649</strong></td><!-- 114 -->
		<td><strong>dist = 111.61</strong><br/>corr = 0.01136</td><!-- 115 -->
	</tr>
	<tr> <!-- 12x -->
		<td><strong>Model 2</strong></td>
		<td><strong>dist = 109.49</strong><br/>corr = 0.00711</td><!-- 120 -->
		<td><strong>dist = 110.73</strong><br/>corr = 0.02130</td><!-- 121 -->
		<td><strong>dist = 109.24</strong><br/>corr = 0.00984</td><!-- 122 -->
		<td><strong>dist = 109.32</strong><br/>corr = 0.00743</td><!-- 123 -->
		<td><strong>dist = 110.73</strong><br/>corr = 0.02188</td><!-- 124 -->
		<td><strong>dist = 109.26</strong><br/>corr = 0.00950</td><!-- 125 -->
	</tr>
	<tr> <!-- 13x -->
		<td><strong>Model 3</strong></td>
		<td>dist = 113.37<br/>corr = 0.02381</td><!-- 130 -->
		<td>dist = 113.37<br/>corr = 0.02381</td><!-- 131 -->
		<td><strong>dist = 113.35</strong><br/>corr = 0.02381</td><!-- 132 -->
		<td>dist = 113.37<br/>corr = 0.02381</td><!-- 133 -->
		<td>dist = 113.37<br/>corr = 0.02381</td><!-- 134 -->
		<td><strong>dist = 113.35</strong><br/>corr = 0.02381</td><!-- 135 -->
	</tr>
	<tr> <!-- 14x -->
		<td><strong>Model 4</strong></td>
		<td><strong>dist = 109.83</strong><br/>corr = 0.01019</td><!-- 140 -->
		<td><strong>dist = 111.56</strong><br/>corr = 0.02239</td><!-- 141 -->
		<td><strong>dist = 111.43</strong><br/>corr = 0.01767</td><!-- 142 -->
		<td><strong>dist = 109.70</strong><br/>corr = 0.00911</td><!-- 143 -->
		<td><strong>dist = 111.69</strong><br/><strong>corr = 0.02511</strong></td><!-- 144 -->
		<td><strong>dist = 111.42</strong><br/>corr = 0.01778</td><!-- 145 -->
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
		<td>dist = 114.13<br/>corr = 0.00569</td><!-- 200 -->
		<td>dist = 114.02<br/>corr = 0.00765</td><!-- 201 -->
		<td>dist = 113.61<br/>corr = 0.01460</td><!-- 202 -->
		<td>dist = 114.11<br/>corr = 0.00601</td><!-- 203 -->
		<td>dist = 114.02<br/>corr = 0.00765</td><!-- 204 -->
		<td>dist = 113.61<br/>corr = 0.01463</td><!-- 205 -->
	</tr>
	<tr> <!-- 21x -->
		<td><strong>Model 1</strong></td>
		<td></td>
		<td><strong>dist = 111.76</strong><br/>corr = 0.00493</td><!-- 211 -->
		<td><strong>dist = 108.68</strong><br/>corr = 0.00536</td><!-- 212 -->
		<td></td>
		<td><strong>dist = 111.76</strong><br/>corr = 0.00493</td><!-- 214 -->
		<td><strong>dist = 108.68</strong><br/>corr = 0.00537</td><!-- 215 -->
	</tr>
	<tr> <!-- 22x -->
		<td><strong>Model 2</strong></td>
		<td></td>
		<td></td>
		<td><strong>dist = 112.84</strong><br/>corr = 0.00451</td><!-- 222 -->
		<td></td>
		<td></td>
		<td><strong>dist = 112.90</strong><br/>corr = 0.00451</td><!-- 225 -->
	</tr>
	<tr> <!-- 23x -->
		<td><strong>Model 3</strong></td>
		<td>dist = 113.37<br/>corr = 0.02381</td><!-- 230 -->
		<td>dist = 113.37<br/>corr = 0.02381</td><!-- 231 -->
		<td>dist = 113.37<br/>corr = 0.02381</td><!-- 232 -->
		<td>dist = 113.37<br/>corr = 0.02381</td><!-- 233 -->
		<td>dist = 113.37<br/>corr = 0.02381</td><!-- 234 -->
		<td>dist = 113.37<br/>corr = 0.02381</td><!-- 235 -->
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
