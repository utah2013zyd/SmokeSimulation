#define IX(i,j) ((i)+(N+2)*(j))
#define WARPSIZE 32
/*__global__ void inner_set_bnd(int N, int b, float* x)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	device_set_bnd(N, b, x, i, j);

	if(i <= N)
	{
		float x1 = x[IX(1, i)], x2 = x[IX(N, i)], x3 = x[IX(i, 1)], x4 = x[IX(i, N)];
		x[IX(0,   i)] = b==1 ? -x1 : x1;
		x[IX(N+1, i)] = b==1 ? -x2 : x2;
		x[IX(i,   0)] = b==2 ? -x3 : x3;
		x[IX(i, N+1)] = b==2 ? -x4 : x4;
	}
	__syncthreads();
	int warp = i/WARPSIZE; 
	int offset = i%WARPSIZE;
	if(warp == 0 && offset == 0)
		x[IX(0, 0)] = 0.5f*(x[IX(1,0)] + x[IX(0, 1)]);
	else if(warp == 1 && offset == 0)
		x[IX(0, N+1)] = 0.5*(x[IX(1, N+1)] + x[IX(0, N)]);
	else if(warp == 2 && offset == 0)
		x[IX(N+1,0)] = 0.5*(x[IX(N, 0)] + x[IX(N+1, 1)]);
	else if(warp == 3 && offset == 0)
		x[IX(N+1,N+1)] = 0.5*(x[IX(N, N+1)] + x[IX(N+1, N)]);

}*/
__device__  void device_set_bnd(int N, int b, float* x, int i, int j)
{
	if(i == 0)
		x[IX(i, j)] = b==1 ? -x[IX(i+1, j)]:x[IX(i+1, j)];
	if(i == N+1)	
		x[IX(i, j)] = b==1 ? -x[IX(i-1, j)]:x[IX(i-1, j)];
	if(j == 0)
		x[IX(i, j)] = b==2 ? -x[IX(i, j+1)]:x[IX(i, j+1)];
	if(j == N+1)
		x[IX(i, j)] = b==2 ? -x[IX(i, j-1)]:x[IX(i, j-1)];

	if(i == 0 && j == 0)
		x[IX(i, j)] = 0.5*(x[IX(i+1, j)] + x[IX(i, j+1)]);
	if(i == 0 && j == N+1) 
		x[IX(i, j)] = 0.5*(x[IX(i+1, j)] + x[IX(i, j-1)]);
	if(i == N+1 && j == 0)
		x[IX(i, j)] = 0.5*(x[IX(i-1, j)] + x[IX(i, j+1)]);
	if(i == N+1 && j == N+1)
		x[IX(i, j)] = 0.5*(x[IX(i-1, j)] + x[IX(i, j-1)]);
}

__global__ void inner_set_bnd(int N, int b, float* x)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	device_set_bnd(N, b, x, i, j);
}
__host__ void cuda_set_bnd(int N, int b, float* x)
{
	float* d_x;
	cudaMalloc((void **) &d_x, (N+2)*(N+2)*sizeof(float));
	cudaMemcpy(d_x, x, (N+2)*(N+2)*sizeof(float), cudaMemcpyHostToDevice);
	
	dim3 threadPerBlock(32, 32);
	dim3 blockPerGrid((N+2)/32, (N+2)/32);
	
	inner_set_bnd<<<blockPerGrid, threadPerBlock>>>(N, b, d_x);

	cudaMemcpy(x, d_x, (N+2)*(N+2)*sizeof(float), cudaMemcpyDeviceToHost);
}

__global__ void inner_advect(int N, int b, float* d, float* d0, float* u, float* v, float dt)
{
	int i, j, i0, j0, i1, j1;
	float x, y, s0, t0, s1, t1, dt0;
	
	i = threadIdx.x + blockIdx.x*blockDim.x;
	j = threadIdx.y + blockIdx.y*blockDim.y;
	dt0 = dt*N;

	x = i - dt0*u[IX(i, j)];
	y = j - dt0*v[IX(i, j)];
	if(x < 0.5f)
		x = 0.5f;
	if(x > N+0.5f)
		x = N+0.5f;
	i0 = (int)x; i1 = i0+1;
	if(y < 0.5f)
		y = 0.5f;
	if(y > N+0.5f)
		y = N+0.5f;
	j0 = (int)y; j1 = j0+1;

	s1 = x - i0; s0 = 1 - s1; t1 = y - j0; t0 = 1 - t1;
	d[IX(i, j)] = s0*(t0*d0[IX(i0, j0)] + t1*d0[IX(i0, j1)]) + s1*(t0*d0[IX(i1, j0)] + t1*d0[IX(i1, j1)]);
	
	device_set_bnd(N, b, d, i, j);
}

__host__ void cuda_advect(int N, int b, float* d, float* d0, float* u, float* v, float dt)
{
	float* d_d, * d_d0, * d_u, * d_v;
	cudaMalloc((void **) &d_d, (N+2)*(N+2)*sizeof(float));
	cudaMalloc((void **) &d_d0, (N+2)*(N+2)*sizeof(float));
	cudaMalloc((void **) &d_u, (N+2)*(N+2)*sizeof(float));
	cudaMalloc((void **) &d_v, (N+2)*(N+2)*sizeof(float));

	cudaMemcpy(d_d, d, (N+2)*(N+2)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_d0, d0, (N+2)*(N+2)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_u, u, (N+2)*(N+2)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_v, v, (N+2)*(N+2)*sizeof(float), cudaMemcpyHostToDevice);

	dim3 threadPerBlock(32, 32);
	dim3 blockPerGrid((N+2)/32, (N+2)/32);
	inner_advect<<<threadPerBlock, blockPerGrid>>>(N, b, d_d, d_d0, d_u, d_v, dt);

	cudaMemcpy(d, d_d, (N+2)*(N+2)*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_d);
	cudaFree(d_d0);
	cudaFree(d_u);
	cudaFree(d_v);
}
