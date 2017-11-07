//v2:可以正确计算所有,效率较低,没有使用共享内存
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include <stdlib.h>

int myLU_analysis(double *vU,int *colU,int *rowU,int nnzU,double *vL,int *rowL,int *colL,int nnzL,int N,int *father,int* fatherNUM,int *level,int *levelNUM,int *nnz){
	int *l=(int *)malloc(N*sizeof(int));
	for(int i=0;i<N;i++)
		l[i]=0;
	for(int i=0;i<N;i++){
		//int flag=0;
		for(int j=rowU[i]+1;j<rowU[i+1];j++){
			//if(l[i]+1>l[colU[j]]){
			//	l[colU[j]]=l[i]+1;
			//	fatherNUM[colU[j]]=1;
			//	father[colU[j]*N+fatherNUM[colU[j]]-1]=i;
			//}
			//if((l[i]+1==l[colU[j]])&&(father[colU[j]*N+fatherNUM[colU[j]]-1]!=i)){
			fatherNUM[colU[j]]++;
			father[colU[j]*N+fatherNUM[colU[j]]-1]=i;
			//}
			//if((l[i]+1==l[colU[j]])&&(father[colU[j]]!=i)){
			//	l[i]=l[colU[j]];
			//	father[i]=father[colU[j]];
			//	l[colU[j]]++;
			//	father[colU[j]]=i;
			//	i--;
			//	flag=1;
			//}
		}
		//if(flag) 
		//	continue;
		//for(int j=colL[i]+1;j<colL[i+1];j++){
		//	//if(l[i]+1>l[rowL[j]]){
		//	//	l[rowL[j]]=l[i]+1;
		//	//	fatherNUM[rowL[j]]=1;
		//	//	father[rowL[j]*N+fatherNUM[rowL[j]]-1]=i;
		//	//}
		//	//if((l[i]+1==l[rowL[j]])&&(father[rowL[j]*N+fatherNUM[rowL[j]]-1]!=i)){
		//	fatherNUM[rowL[j]]++;
		//	father[rowL[j]*N+fatherNUM[rowL[j]]-1]=i;
		//	//}
		//	//if((l[i]+1==l[rowL[j]])&&(father[rowL[j]]!=i)){
		//	//	l[i]=l[rowL[j]];
		//	//	father[i]=father[rowL[j]];
		//	//	l[rowL[j]]++;
		//	//	father[rowL[j]]=i;
		//	//	i--;
		//	//	flag=1;
		//	//}
		//}//建立节点依赖关系
		//if(flag) 
		//	continue;
		for(int j=rowU[i]+1;j<rowU[i+1];j++){//产生新的注入元，以0填入
			int nj=colU[j];//欲更新的元素纵坐标
			for(int k=colL[i]+1;k<colL[i+1];k++){
				int ni=rowL[k];//欲更新的元素横坐标
				if(ni>nj){//在L里找nj列非零元
					if(ni>rowL[colL[nj+1]-1]){
						int x=colL[nj+1];
						nnzL++;
						for(int y=nnzL-1;y>x;y--){
							rowL[y]=rowL[y-1];
							vL[y]=vL[y-1];
						}
						rowL[x]=ni;
						vL[x]=0;
						for(int y=nj+1;y<N+1;y++)
							colL[y]++;
					}//在该列最后插入新注入元
					else
						for(int x=colL[nj];x<colL[nj+1]-1;x++)
							if((ni>rowL[x])&&(ni<rowL[x+1])){
								nnzL++;
								for(int y=nnzL-1;y>x+1;y--){
									rowL[y]=rowL[y-1];
									vL[y]=vL[y-1];
								}
								rowL[x+1]=ni;
								vL[x+1]=0;
								for(int y=nj+1;y<N+1;y++)
									colL[y]++;
								break;
							}//在中间插入新注入元
				}
				if(ni<nj){//在U里找ni行非零元
					if(nj>colU[rowU[ni+1]-1]){
						int x=rowU[ni+1];
						nnzU++;
						for(int y=nnzU-1;y>x;y--){
							colU[y]=colU[y-1];
							vU[y]=vU[y-1];
						}
						colU[x]=nj;
						vU[x]=0;
						for(int y=ni+1;y<N+1;y++)
							rowU[y]++;
					}//在该行最后插入新注入元
					else
						for(int x=rowU[ni];x<rowU[ni+1]-1;x++)
							if((nj>colU[x])&&(nj<colU[x+1])){
								nnzU++;
								for(int y=nnzU-1;y>x+1;y--){
									colU[y]=colU[y-1];
									vU[y]=vU[y-1];
								}
								colU[x+1]=nj;
								vU[x+1]=0;
								for(int y=ni+1;y<N+1;y++)
									rowU[y]++;
								break;
							}//在中间插入新注入元
				}	
			}
		}
		//for(int j=0;j<nnzU;j++)
		//	printf("%f ",vU[j]);
		//printf("\n");
		//for(int j=0;j<nnzU;j++)
		//	printf("%d ",colU[j]);
		//printf("\n");
		//for(int j=0;j<N;j++)
		//	printf("%d ",rowU[j]);
		//printf("\n");
	}
	//for(int i=0;i<N;i++)
	//	printf("%d ",l[i]);
	//printf("\n");
	int leveldepth=0;
	for(int i=0;i<N;i++){
		levelNUM[l[i]]++;
		if (l[i]>leveldepth)
			leveldepth=l[i];
	}
	int *start=(int *)malloc(N*sizeof(int));
	start[0]=0;
	for(int i=1;i<=leveldepth;i++)
		start[i]=start[i-1]+levelNUM[i-1];
	int *num=(int *)malloc(N*sizeof(int));
	for(int i=0;i<N;i++)
		num[i]=0;
	for(int i=0;i<N;i++){
		int j=start[l[i]]+num[l[i]];
		level[j]=i;
		num[l[i]]++;
	}
	nnz[0]=nnzU;
	nnz[1]=nnzL;
	free(start);
	free(l);
	return leveldepth;
}
__device__ double MyatomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
			__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}
__global__ void myLU_solve(double *vU,int *colU,int *rowU,double *vL,int *rowL,int *colL,int *father,int *fatherNUM,int *block2node,int *cntblocks,volatile int *flag,volatile int *flag2,int *N){

	int node=block2node[ blockIdx.x];
	/*__shared__ int start;
	start=0;*/
	//printf("%d ",blockIdx.x);
	//int ff=1;
	//if(node==2){
	//for(int i=0;i<1000000;i++);
	// //printf("%d ",flag[0]);
	//}
	//if(node==0) printf("%d ",ff);
	//if(node==3) printf("%d ",node);
	__shared__ int tempcolU[200];
	__shared__ int temprowL[200];//暂时假设每行每列非零元数目都不超过100
	int rows=rowU[node+1]-rowU[node]-1;
	int cols=colL[node+1]-colL[node]-1;
	//if(node==1) printf("%d %d\n",rows,cols);
	int id=threadIdx.x;
	if(id<rows){
		tempcolU[id]=colU[rowU[node]+1+id];
		//if(node==1 && id==1) printf("%d\n",tempcolU[id]);
	}
	if(id<cols)
		temprowL[id]=rowL[colL[node]+1+id];
	__syncthreads();
	//if(node==1) printf("%d\n",tempcolU[1]);
	int rowUnode=rowU[node],colLnode=colL[node];
	
    int threadsID=(blockIdx.x-cntblocks[node]) * blockDim.x + threadIdx.x;
	int index,sign;
	if(threadsID<cols*rows){//更新运算
		int i=threadsID/cols,j=threadsID%cols;
		//if(node==5) printf("%f %f",vU[rowU[node]+1+i],vL[colL[node]+1+j]);
		int ni=temprowL[j],nj=tempcolU[i];
		//if(ni==52 && nj==51)
		//	printf("%d %d\n",node,threadsID);
		if(ni>nj){//更新下三角中的元素
			for(int x=colL[nj]+1;x<colL[nj+1];x++)
				if (rowL[x]==ni){
					sign=0;
					index=x;
					//if(x==0) printf("%d %d %d %d %d\n",node,ni,nj,x,j);
					break;
				}
		}
		else{
			for(int x=rowU[ni];x<rowU[ni+1];x++)
				if (colU[x]==nj){
					sign=1;
					index=x;
					//if(node==5) printf("%f ",val);
					//if(threadsID==1) printf("%d %d %d %d\n",node,ni,nj,x);
					break;
				}
		}
	}

	//__syncthreads();
	while(1){
		//if(node==2)
		//  printf("%d %d\n",kk,flag[0]);
		//printf("ff=fd");
		//if (node==2) for(int i=0;i<10000;i++);
		//if(threadIdx.x<=255){
		int ff=1;

		for(int i=0;i<fatherNUM[node];i++){
			int fanode=father[node*(*N)+i];
			//if(node==51) printf("%d %d\n",fatherNUM[node],fanode);
			//if(node==2) printf("%d ",fanode);
			int limit=(rowU[fanode+1]-rowU[fanode]-1)*(colL[fanode+1]-colL[fanode]-1);
			//if(node==2) printf("%d %d",fanode,limit);
			if(flag[fanode]!=limit){
				ff=0;
				break;
			}
		}

		//start=ff;
		//printf("start=%d ",start);
		//}
		//if(node==3) printf("ff=%d ",ff);
		if(ff)
			break;
	}

	//if(node==7)
	//	printf("%d %d\n",rowU[7],colL[7]);
	if(threadsID<cols){//规格化运算
		//if (node==3) printf("%f %f\n",vL[colL[node]+1+threadsID],vU[rowU[node]]);
		double para=1/vU[rowU[node]];
		vL[colLnode+1+threadsID]=vL[colLnode+1+threadsID]*para;
		//if (node==3) printf("%d %f\n",colL[node]+1+threadsID,vL[colL[node]+1+threadsID]);
		//if(node==51) printf("%f %f\n",vL[colLnode+1+threadsID],vU[rowU[node]]);
		atomicAdd(&(int)flag2[node],1);
	}		
	while(1){
		if(flag2[node]==cols) break;
	}
	
	//if (node==2) printf("%f ",vL[4]);
	if(threadsID<cols*rows){//更新运算
		int i=threadsID/cols,j=threadsID%cols;
		double val=-vU[rowUnode+1+i]*vL[colLnode+1+j];
		//if(val==0) printf("%d %d %d %d\n",node,threadsID,i,j);
		//if(node==32 && threadsID==6) printf("%f %f %f\n",vU[rowUnode+1+i],vL[colLnode+1+j],val);
		if(!sign){
			MyatomicAdd(&vL[index],val);
			//if(index[threadsID]==0) printf("%d %d\n",node,threadsID);
		}
		else{
			MyatomicAdd(&vU[index],val);
		}			
		atomicAdd(&(int)flag[node],1);			
	}
	//if(node==51) printf("%f\n",vU[rowU[node]]);
	//flag[node]=1;

	//if(node==2) printf("%d %d ",flag[node]);
	//if(node==6) printf("aaavv ");
}
int main()
{
    FILE *in;
	if((in=fopen("C:\\home\\wmx\\myLU\\input\\J39.txt","rt+"))==NULL){
		printf("Cannot open file strike any key exit!");
		getchar();
		exit(1);
	}
	int N,nnzU,nnzL;
	fscanf(in,"%d",&N);
	fscanf(in,"%d",&nnzU);
	double *vU=(double *)malloc(5*nnzU*sizeof(double));
	int *colU=(int *)malloc(5*nnzU*sizeof(int));
	int *rowU=(int *)malloc((N+1)*sizeof(int));
	for(int i=0;i<nnzU;i++)
		fscanf(in,"%lf",&vU[i]);
	for(int i=0;i<nnzU;i++)
		fscanf(in,"%d",&colU[i]);
	for(int i=0;i<N+1;i++)
		fscanf(in,"%d",&rowU[i]);

	fscanf(in,"%d",&nnzL);
	double *vL=(double *)malloc(5*nnzL*sizeof(double));
	int *rowL=(int *)malloc(5*nnzL*sizeof(int));
	int *colL=(int *)malloc((N+1)*sizeof(int));
	for(int i=0;i<nnzL;i++)
		fscanf(in,"%lf",&vL[i]);
	for(int i=0;i<nnzL;i++)
		fscanf(in,"%d",&rowL[i]);
	for(int i=0;i<N+1;i++)
		fscanf(in,"%d",&colL[i]);
	fclose(in);
	int *level=(int *)malloc(N*sizeof(int));
	int *levelNUM=(int *)malloc(N*sizeof(int));
	for(int i=0;i<N;i++)
		levelNUM[i]=0;
	int *nnz=(int *)malloc(2*sizeof(int));
	int *father=(int *)malloc(N*N*sizeof(int));
	for(int i=0;i<N*N;i++)
		father[i]=-1;
	int *fatherNUM=(int *)malloc(N*sizeof(int));
	for(int i=0;i<N;i++)
		fatherNUM[i]=0;
	int leveldepth=myLU_analysis(vU,colU,rowU,nnzU,vL,rowL,colL,nnzL,N,father,fatherNUM,level,levelNUM,nnz);
	nnzU=nnz[0];
	nnzL=nnz[1];
	//for(int i=0;i<N;i++)
	//	printf("%d ",level[i]);
	//printf("\n");
	//for(int i=0;i<=leveldepth;i++)
	//	printf("%d ",levelNUM[i]);
	//printf("\n");
	//for(int i=0;i<N;i++){
	//	for(int j=0;j<fatherNUM[i];j++)
	//	    printf("%d ",father[i*N+j]);
	//	printf("\n");
	//}
	//for(int i=0;i<N;i++){
	//	printf("%d ",fatherNUM[i]);
	//}
	//printf("%d\n",fatherNUM[51]);
	int *threadsNeed=(int *)malloc(N*sizeof(int));
	int *blocksNeed=(int *)malloc(N*sizeof(int));
	int *cntblocks=(int *)malloc(N*sizeof(int));
	cntblocks[0]=0;
	int threadsperblock=256,totalblocks=0;
	for(int i=0;i<N;i++){
		int rows=rowU[i+1]-rowU[i]-1;
		int cols=colL[i+1]-colL[i]-1;
		threadsNeed[i]=rows*cols;
		blocksNeed[i]=threadsNeed[i]/threadsperblock+1;
		totalblocks+=blocksNeed[i];
		if(i>0)
			cntblocks[i]=cntblocks[i-1]+blocksNeed[i-1];
	}
	//for(int i=0;i<N;i++){
	//	printf("%d %d\n",blocksNeed[i],cntblocks[i]);
	//}
	int *block2node=(int *)malloc(totalblocks*sizeof(int));
	for(int i=0;i<N;i++){
		for(int j=cntblocks[i];j<cntblocks[i]+blocksNeed[i];j++)
			block2node[j]=i;
	}
	//for(int i=0;i<totalblocks;i++)
	//	printf("%d ",block2node[i]);
	int *flag=(int *)malloc(N*sizeof(int));//全局标记位
	for(int i=0;i<N;i++)
		flag[i]=0;

	double *d_vU,*d_vL;
	cudaMalloc((void**)&d_vU,nnzU* sizeof(double));
	cudaMemcpy(d_vU,vU,nnzU* sizeof(double), cudaMemcpyHostToDevice); 
	cudaMalloc((void**)&d_vL,nnzL* sizeof(double));
	cudaMemcpy(d_vL,vL,nnzL* sizeof(double), cudaMemcpyHostToDevice); 

	int *d_colL,*d_rowL,*d_rowU,*d_colU;
	cudaMalloc((void**)&d_colU,nnzU* sizeof(int));
	cudaMemcpy(d_colU,colU,nnzU* sizeof(int), cudaMemcpyHostToDevice); 
	cudaMalloc((void**)&d_rowU,(N+1)* sizeof(int));
	cudaMemcpy(d_rowU,rowU,(N+1)* sizeof(int), cudaMemcpyHostToDevice); 
	cudaMalloc((void**)&d_colL,(N+1)* sizeof(int));
	cudaMemcpy(d_colL,colL,(N+1)* sizeof(int), cudaMemcpyHostToDevice); 
	cudaMalloc((void**)&d_rowL,nnzL* sizeof(int));
	cudaMemcpy(d_rowL,rowL,nnzL* sizeof(int), cudaMemcpyHostToDevice); 

	int *d_father,*d_fatherNUM,*d_block2node,*d_cntblocks,*d_flag,*d_flag2;
	cudaMalloc((void**)&d_father,N*N* sizeof(int));
	cudaMemcpy(d_father,father,N*N* sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_fatherNUM,N* sizeof(int));
	cudaMemcpy(d_fatherNUM,fatherNUM,N* sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_block2node,totalblocks* sizeof(int));
	cudaMemcpy(d_block2node,block2node,totalblocks* sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_cntblocks,N* sizeof(int));
	cudaMemcpy(d_cntblocks,cntblocks,N* sizeof(int), cudaMemcpyHostToDevice);
	//cudaError_t err2 = cudaGetLastError();
	cudaMalloc((void**)&d_flag,N* sizeof(int));
	cudaMemcpy(d_flag,flag,N* sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_flag2,N* sizeof(int));
	cudaMemcpy(d_flag2,flag,N* sizeof(int), cudaMemcpyHostToDevice);

	int *d_N;
	cudaMalloc((void**)&d_N,sizeof(int));
	cudaMemcpy(d_N,&N,sizeof(int), cudaMemcpyHostToDevice);

	myLU_solve<<<totalblocks,threadsperblock>>>(d_vU,d_colU,d_rowU,d_vL,d_rowL,d_colL,d_father,d_fatherNUM,d_block2node,d_cntblocks,d_flag,d_flag2,d_N);
	cudaThreadSynchronize();
	cudaError_t err = cudaGetLastError();
	cudaMemcpy(vU,d_vU,nnzU* sizeof(double), cudaMemcpyDeviceToHost); 
/*	cudaMemcpy(colU,d_colU,nnzU* sizeof(double), cudaMemcpyDeviceToHost); 
	cudaMemcpy(rowU,d_rowU,(N+1)* sizeof(double), cudaMemcpyDeviceToHost);*/ 
	cudaMemcpy(vL,d_vL,nnzL* sizeof(double), cudaMemcpyDeviceToHost); 
	//cudaMemcpy(rowL,d_rowL,nnzL* sizeof(double), cudaMemcpyDeviceToHost); 
	//cudaMemcpy(colL,d_colL,(N+1)* sizeof(double), cudaMemcpyDeviceToHost); 
	cudaMemcpy(flag,d_flag,N* sizeof(int), cudaMemcpyDeviceToHost); 
	//printf("%f\n",vU[rowU[51]]);
	FILE *outfp;
	outfp=fopen("C:\\home\\wmx\\myLU\\output\\J39LU2.txt","wt+");
	fprintf(outfp,"%d\n",N);
	fprintf(outfp,"%d\n",nnzU);
	for(int i=0;i<nnzU;i++)
		fprintf(outfp,"%f ",vU[i]);
	fprintf(outfp,"\n");
	for(int i=0;i<nnzU;i++)
		fprintf(outfp,"%d ",colU[i]);
	fprintf(outfp,"\n");
	for(int i=0;i<(N+1);i++)
		fprintf(outfp,"%d ",rowU[i]);
	fprintf(outfp,"\n");
	fprintf(outfp,"%d\n",nnzL);
	for(int i=0;i<nnzL;i++)
		fprintf(outfp,"%f ",vL[i]);
	fprintf(outfp,"\n");
	for(int i=0;i<nnzL;i++)
		fprintf(outfp,"%d ",rowL[i]);
	fprintf(outfp,"\n");
	for(int i=0;i<(N+1);i++)
		fprintf(outfp,"%d ",colL[i]);
	fprintf(outfp,"\n");
	//printf("\n");
	//for(int i=0;i<N;i++)
	//	printf("%d ",flag[i]);
	fclose(outfp);
    return 0;
}



