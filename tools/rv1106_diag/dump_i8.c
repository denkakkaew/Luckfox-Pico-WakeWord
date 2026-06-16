#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rknn_api.h"
int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"rb");fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);
  void*m=malloc(n);if(fread(m,1,n,f)!=(size_t)n)return 2;fclose(f);
  rknn_context ctx;if(rknn_init(&ctx,m,n,0,NULL))return 1;
  rknn_tensor_attr ia,oa;memset(&ia,0,sizeof(ia));ia.index=0;rknn_query(ctx,RKNN_QUERY_INPUT_ATTR,&ia,sizeof(ia));
  memset(&oa,0,sizeof(oa));oa.index=0;rknn_query(ctx,RKNN_QUERY_OUTPUT_ATTR,&oa,sizeof(oa));
  rknn_tensor_mem*im=rknn_create_mem(ctx,ia.size_with_stride);
  rknn_tensor_mem*om=rknn_create_mem(ctx,oa.size_with_stride);
  memset(im->virt_addr,0,ia.size_with_stride);
  rknn_set_io_mem(ctx,im,&ia); rknn_set_io_mem(ctx,om,&oa);
  int H=ia.dims[1],W=ia.dims[2],ws=ia.w_stride?ia.w_stride:W;
  float*spec=malloc((size_t)H*W*sizeof(float));
  FILE*sf=fopen(argv[2],"rb");fread(spec,sizeof(float),(size_t)H*W,sf);fclose(sf);
  int8_t*in=(int8_t*)im->virt_addr;
  for(int h=0;h<H;h++)for(int w=0;w<W;w++){int q=(int)lrintf(spec[h*W+w]/ia.scale)+ia.zp;
    if(q<-128)q=-128;if(q>127)q=127; in[h*ws+w]=(int8_t)q;}
  if(rknn_run(ctx,NULL)){printf("run failed\n");return 1;}
  int N=oa.n_elems; int8_t*o=(int8_t*)om->virt_addr;
  float*buf=malloc(N*sizeof(float));
  for(int k=0;k<N;k++) buf[k]=((int)o[k]-oa.zp)*oa.scale;
  FILE*of=fopen(argv[3],"wb"); fwrite(buf,sizeof(float),N,of); fclose(of);
  printf("n_elems=%d fmt=%d type=%d zp=%d scale=%f\n",N,oa.fmt,oa.type,oa.zp,oa.scale);
  return 0;
}
