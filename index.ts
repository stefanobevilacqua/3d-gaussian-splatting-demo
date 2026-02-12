type Vec3 = [number, number, number]
type Model = {
    vertices: Vec3[],
    faces: Vec3[],
    normals: Vec3[]
}
type Mat3 = [Vec3, Vec3, Vec3]
type GaussianSplat = {
    position: Vec3,
    rotation: Mat3,
    scale: Vec3,
    color: Vec3,
    opacity: number,
}

function vec3_add(a: Vec3, b: Vec3): Vec3 {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

function vec3_sub(a: Vec3, b: Vec3): Vec3 {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

function vec3_cross(a: Vec3, b: Vec3): Vec3 {
    return [
	a[1]*b[2] - a[2]*b[1],
	a[2]*b[0] - a[0]*b[2],
	a[0]*b[1] - a[1]*b[0],
    ]
}

function vec3_dot(a: Vec3, b: Vec3): number {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

function vec3_module(v: Vec3): number {
    return Math.sqrt(vec3_dot(v, v))
}

function vec3_normalize(v: Vec3): Vec3 {
    return vec3_scale(v, 1.0 / vec3_module(v))
}

function vec3_scale(v: Vec3, k: number): Vec3 {
    return [v[0]*k, v[1]*k, v[2]*k]
}

function mat3_mult_vec3(m: Mat3, v: Vec3) {
    let r: Vec3 = [0, 0, 0]
    for(let i =0; i < 3; i++) {
	r = vec3_add(r, vec3_scale(m[i]!, v[i]!))
    }
    return r;
}

async function getBunnyModel(): Promise<Model> {
    const text = await fetch("bunny.obj").then(response => response.text())
    let model: Model = {vertices: [], faces: [], normals: []}
    text.split(/\r\n|\n|\r/).forEach(line => {
	if(line.startsWith("v")) {
	    model.vertices.push(line.split(/\s+/).slice(1).map(x => parseFloat(x)) as Vec3)
	}
	if(line.startsWith("f")) {
	    let face = line.split(/\s+/).slice(1).map(x => parseInt(x) - 1) as Vec3 
	    model.faces.push(face)
	    let triangle = [
		model.vertices[face[0]],
		model.vertices[face[1]],
		model.vertices[face[2]],
	    ]
	    model.normals.push(vec3_cross(vec3_sub(triangle[1]!, triangle[0]!),
					  vec3_sub(triangle[2]!, triangle[0]!)))
	}
    })
    return model
}

function generateGaussianSplats(model: Model, numSplats: number): GaussianSplat[] {
    let areas: number[] = model.normals.map((normal: Vec3) => {
	return vec3_module(normal) // No need to multiply by 0.5 here
    })
    let totalArea = areas.reduce((total, area) => { return total + area })
    let numSplatsPerFace = areas.map(area => Math.round(area * numSplats / totalArea))
    let splats: GaussianSplat[] = []
    for(let i = 0; i < model.faces.length; i++) {
	let face = model.faces[i]!
	let triangle = [
	    model.vertices[face[0]],
	    model.vertices[face[1]],
	    model.vertices[face[2]],
	]
	for(let j = 0; j < numSplatsPerFace[i]!; j++) {
	    let u = Math.random()
	    let v = Math.random()
	    if(u + v > 1) {
		u = 1 - u
		v = 1 - v
	    }
	    let position = vec3_scale(triangle[0]!, 1-u-v)
	    position = vec3_add(position, vec3_scale(triangle[1]!, u))
	    position = vec3_add(position, vec3_scale(triangle[2]!, v))
	    splats.push({
		position: position,
		// TODO: get real rotation matrix from normal
		rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
		scale: [0.01, 0.01, 0.01],
		color: [0.9, 0.9, 0.9],
		opacity: 0.01
	    })
	}
    }
    return splats
}

type RenderContext = {
    canvas: HTMLCanvasElement,
    context: GPUCanvasContext,
    device: GPUDevice,
    splatsBuffer: GPUBuffer,
    pipeline: GPURenderPipeline,
    bindGroup: GPUBindGroup,
    numOfSplats: number,
    dt: number,
}

function uploadSplatsToGPU(device: GPUDevice, splats: GaussianSplat[]): GPUBuffer {
    const cpuBuffer: number[] = []
    splats.forEach(splat => {
	cpuBuffer.push(splat.position[0])
	cpuBuffer.push(splat.position[1])
	cpuBuffer.push(splat.position[2])
	cpuBuffer.push(0.0)	// Padding
	cpuBuffer.push(splat.rotation[0][0])
	cpuBuffer.push(splat.rotation[0][1])
	cpuBuffer.push(splat.rotation[0][2])
	cpuBuffer.push(0.0)
	cpuBuffer.push(splat.rotation[1][0])
	cpuBuffer.push(splat.rotation[1][1])
	cpuBuffer.push(splat.rotation[1][2])
	cpuBuffer.push(0.0)
	cpuBuffer.push(splat.rotation[2][0])
	cpuBuffer.push(splat.rotation[2][1])
	cpuBuffer.push(splat.rotation[2][2])
	cpuBuffer.push(0.0)
	cpuBuffer.push(splat.scale[0])
	cpuBuffer.push(splat.scale[1])
	cpuBuffer.push(splat.scale[2])
	cpuBuffer.push(0.0)
	cpuBuffer.push(splat.color[0])
	cpuBuffer.push(splat.color[1])
	cpuBuffer.push(splat.color[2])
	cpuBuffer.push(splat.opacity)
    })
    const cpuBufferView = new Float32Array(cpuBuffer)
    const buffer = device.createBuffer({
	size: cpuBufferView.length * 4,
	usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    })
    device.queue.writeBuffer(buffer, 0, cpuBufferView.buffer)
    return buffer
}

function onFrame(ctx: RenderContext) {
    ctx.canvas.width = ctx.canvas.clientWidth
    ctx.canvas.height = ctx.canvas.clientHeight
    
    let cmd = ctx.device.createCommandEncoder()
    let pass = cmd.beginRenderPass({
	colorAttachments: [{
	    loadOp: "clear",
	    storeOp: "store",
	    view: ctx.context.getCurrentTexture().createView(),
	    clearValue: [0.1, 0.2, 0.3, 1.0]
	}]
    })
    pass.setPipeline(ctx.pipeline)
    pass.setBindGroup(0, ctx.bindGroup)
    pass.draw(4, ctx.numOfSplats)
    pass.end()
    ctx.device.queue.submit([cmd.finish()])
    requestAnimationFrame((dt: number) => onFrame({...ctx, dt: dt}))
}

async function onLoad() {
    const adapter = await navigator.gpu?.requestAdapter()
    const device = await adapter?.requestDevice()
    if(!device) {
	console.error("WebGPU is not supported")
	return
    }

    const canvas = document.getElementById("canvas") as HTMLCanvasElement
    const ctx = canvas.getContext("webgpu") as GPUCanvasContext;
    ctx.configure({
	device: device,
	format: navigator.gpu.getPreferredCanvasFormat()
    })

    const model = await getBunnyModel()
    const splats = generateGaussianSplats(model, 100000)
    
    // tmp
    window.model = model
    window.splats = splats

    const splatsBuffer = uploadSplatsToGPU(device, splats)
    const shaderModule = device.createShaderModule({
	code: await fetch("shader.wgsl").then(r => r.text()),
    })
    const pipeline = device.createRenderPipeline({
	layout: "auto",
	vertex: {
	    module: shaderModule,
	    entryPoint: "vs_main",
	},
	primitive: {
	    topology: "triangle-strip"
	},
	fragment: {
	    module: shaderModule,
	    entryPoint: "fs_main",
	    targets: [{
		format: navigator.gpu.getPreferredCanvasFormat(),
		blend: {
		    alpha: {},
		    color: {
			srcFactor: "src-alpha",
			dstFactor: "one-minus-src-alpha",
			operation: "add"
		    }
		}
	    }]
	}
    })

    const bindGroup = device.createBindGroup({
	layout: pipeline.getBindGroupLayout(0),
	entries: [{
	    binding: 0,
	    resource: splatsBuffer,
	}]
    })

    requestAnimationFrame((dt: number) => onFrame({
	canvas: canvas,
	context: ctx,
	device: device,
	splatsBuffer: splatsBuffer,
	pipeline: pipeline,
	bindGroup: bindGroup,
	numOfSplats: splats.length,
	dt: dt
    }))
}

window.addEventListener("DOMContentLoaded", onLoad);
export {}
