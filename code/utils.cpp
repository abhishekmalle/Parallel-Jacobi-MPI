/**
 * @file    utils.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements common utility/helper functions.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
#include "utils.h"
#include <iostream>
#include <cmath>


// calculate D and R
void cal_DR(const double* A, const int n, double* invD, double* R){
	for (int i=0; i<n; i++)
	{
		invD[i] = 1./A[n*i+i];
		for (int j=0; j<n; j++)
		{
			if (i!=j)
				R[n*i+j] = A[n*i+j];
		}
	}
}


// calculate delta = Ax-b, stores the results in n-D vector delta
void vec_subtract(double* y, const double* b, const int n)
{
	for (int i=0; i<n; i++)
		y[i] = b[i] - y[i]; 
}


// calculate l2-norm of a vector
double L2_norm(const int n, const double* x)
{
	double sum=0;
	for(int i=0;i<n;i++)
		sum += x[i]*x[i];
	return sqrt(sum);
}

// ...

