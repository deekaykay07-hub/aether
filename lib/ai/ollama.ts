/* Aether AI — Ollama Client (Optimized for Small Models) */

export interface AetherAIRequest {
  task: 'resume_analysis' | 'cover_letter' | 'job_match' | 'remote_readiness' | 'skill_gap' | 'interview_feedback';
  input: Record<string, any>;
  options?: {
    temperature?: number;
    maxTokens?: number;
  };
}

export interface AetherAIResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  model: string;
  task: string;
  usedLocalModel: boolean;
}

const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434';
const DEFAULT_MODEL = process.env.OLLAMA_MODEL || 'qwen2.5:1.5b';

async function callOllama(systemPrompt: string, userPrompt: string, options: { temperature?: number; maxTokens?: number } = {}) {
  const model = DEFAULT_MODEL;
  const temperature = options.temperature ?? 0.2;
  const maxTokens = options.maxTokens ?? 600;

  const payload = {
    model,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt },
    ],
    stream: false,
    options: { temperature, num_predict: maxTokens, top_p: 0.9 },
    format: 'json',
  };

  const res = await fetch(`${OLLAMA_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) throw new Error(`Ollama request failed: ${res.status}`);

  const data = await res.json();
  let content = (data.message?.content || data.response || '').trim();
  content = content.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim();

  return { content, model };
}

export async function aetherAI<T = any>(req: AetherAIRequest): Promise<AetherAIResponse<T>> {
  try {
    const { systemPrompt, userPrompt } = await buildPrompts(req.task, req.input);
    const { content, model } = await callOllama(systemPrompt, userPrompt, req.options);
    let parsed;
    try { parsed = JSON.parse(content); }
    catch {
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (jsonMatch) parsed = JSON.parse(jsonMatch[0]);
      else throw new Error('Model did not return valid JSON');
    }
    return { success: true, data: parsed as T, model, task: req.task, usedLocalModel: true };
  } catch (error: any) {
    return { success: false, error: error.message, model: DEFAULT_MODEL, task: req.task, usedLocalModel: true };
  }
}

async function buildPrompts(task: string, input: Record<string, any>) {
  const { getSystemPrompt, buildUserPrompt } = await import('./prompts');
  return {
    systemPrompt: getSystemPrompt(task as any),
    userPrompt: buildUserPrompt(task as any, input),
  };
}
